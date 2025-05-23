import logging
import os
import pickle
import shutil
import struct
import subprocess
import time
from pathlib import Path
from time import perf_counter, sleep, strftime
from typing import Tuple

import fitTools.quasiparticleFunctions as qp
import h5py
import Labber
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv, set_key
from fitTools.utilities import Watt2dBm, dBm2Watt
from matplotlib.colors import TwoSlopeNorm
from pandas import DataFrame
from resonator import reflection, see
from scipy.optimize import curve_fit
from srsinst.dc205 import DC205
from tqdm import tqdm
from VISAdrivers.continuousAlazar import ADC

from flux_fit import find_mapped_resonance

# Global instrument variables
VNA = None
DA = None
SMU = None
LO = None
Drive = None
vs = None
lfVNA = None
FIG_PATH = None
PATH_TO_EXE = None
SPATH = None
device_name = None
TWPA_PUMP = None
SA = None

def scale_voltage_range(old_range: Tuple[float, float], 
                        old_resistance: float, 
                        new_resistance: float, 
                        invariant: str = 'power') -> Tuple[float, float]:
    """
    Scale voltage sweep range according to change in resistance.

    Parameters
    ----------
    old_range : Tuple[float, float]
        (min_voltage, max_voltage) in volts.
    old_resistance : float
        Original resistance in ohms.
    new_resistance : float
        New resistance in ohms.
    invariant : str
        What to preserve: 'current', 'power' (or 'energy').

    Returns
    -------
    Tuple[float, float]
        New (min_voltage, max_voltage) in volts.
    """
    if invariant == 'current':
        scale_factor = new_resistance / old_resistance
    elif invariant in ['power', 'energy']:
        scale_factor = np.sqrt(new_resistance / old_resistance)
    else:
        raise ValueError("invariant must be one of: 'current', 'power', 'energy'")

    min_v, max_v = old_range
    return min_v * scale_factor, max_v * scale_factor


def scale_sweep_rate(old_rate: float,
                     old_resistance: float,
                     new_resistance: float,
                     invariant: str = 'power') -> float:
    """
    Scale voltage sweep rate according to change in resistance.

    Parameters
    ----------
    old_rate : float
        Original sweep rate in volts/sec.
    old_resistance : float
        Original resistance in ohms.
    new_resistance : float
        New resistance in ohms.
    invariant : str
        What to preserve: 'current', 'power' (or 'energy').

    Returns
    -------
    float
        New sweep rate in volts/sec.
    """
    if invariant == 'current':
        scale_factor = new_resistance / old_resistance
    elif invariant in ['power', 'energy']:
        scale_factor = np.sqrt(new_resistance / old_resistance)
    else:
        raise ValueError("invariant must be one of: 'current', 'power', 'energy'")

    return old_rate * scale_factor


def initialize_instruments(vna, da=None, smu=None, lo=None, drive=None, srs=None, twpa_pump=None, sa=None):
    """
    Initialize global instrument variables for use in other functions.
    
    Parameters:
    -----------
    vna : instrument object
        Vector Network Analyzer instrument
    da : instrument object, optional
        Digital Attenuator instrument
    smu : instrument object, optional
        Source Meter Unit instrument
    lo : instrument object, optional
        Local Oscillator instrument
    drive : instrument object, optional
        Drive Signal Generator instrument
    srs : instrument object, optional
        SRS instrument for flux biasing
    """
    global VNA, DA, SMU, LO, Drive, vs, TWPA_PUMP, SA
    VNA = vna
    DA = da
    SMU = smu
    LO = lo
    Drive = drive
    vs = srs
    TWPA_PUMP = twpa_pump
    SA = sa
    
def initializeLabberlogging(lfvna):
    global lfVNA
    lfVNA = lfvna

def update_env_variables(spath, project_name, device_name, globus_endpoint=None, globus_token=None):
    """
    Updates or creates .env file with storage paths and Globus credentials.
    
    Parameters:
    -----------
    spath : str
        Local storage path
    project_name : str
        Name of the project
    device_name : str
        Name of the device being measured
    globus_endpoint : str, optional
        Globus endpoint ID
    globus_token : str, optional
        Globus authentication token
    """
    env_path = Path('.env')
    
    # Create .env if it doesn't exist
    if not env_path.exists():
        env_path.touch()
    
    # Update environment variables
    set_key(env_path, 'LOCAL_STORAGE_PATH', spath)
    project_name = project_name + "_" + time.strftime("%m%d%y")
    set_key(env_path, 'PROJECT_NAME', project_name)
    set_key(env_path, 'DEVICE_NAME', device_name)

    # Set default Globus variables if not provided
    if globus_endpoint is None:
        globus_endpoint = "your-globus-endpoint-id"
    if globus_token is None:
        globus_token = "your-globus-token"
        
    set_key(env_path, 'GLOBUS_SOURCE_ENDPOINT', globus_endpoint)
    set_key(env_path, 'GLOBUS_DEST_ENDPOINT', globus_endpoint)
    set_key(env_path, 'GLOBUS_TOKEN', globus_token)
    set_key(env_path, 'HPC_BASE_PATH', '/path/on/hpc/storage')
    
    # Reload environment variables
    load_dotenv()

def initialize_logging(lfvna, spath, path_to_exe, fig_path, project_name, d_name):
    global SPATH, PATH_TO_EXE, FIG_PATH, device_name
    SPATH = spath
    PATH_TO_EXE = path_to_exe
    device_name = d_name
    FIG_PATH = fig_path
    initializeLabberlogging(lfvna)
    
    # Update environment variables with project and device info
    update_env_variables(
        spath=spath,
        project_name=project_name,  # You might want to make this configurable
        device_name=device_name
    )

def set_project(base_path, sub_dir=None):
    project_path = os.path.join(base_path, time.strftime("%m%d%y"))
    if sub_dir:
        project_path = os.path.join(project_path, sub_dir)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    print(f"Project path set to: {project_path}")
    return project_path


def bin2csv(filename, saveto = "E:/rawData/"):
    '''Converts the binary data file as input and returns a .csv file with usable data. - JF '''
    drive, path = os.path.splitdrive(filename)
    path,file = os.path.split(path)
    newfile = saveto + file.strip('.bin') + ".csv"
    DATA = np.fromfile(filename,dtype=np.uint16)
    DATA = (DATA - 2047.5) * 0.4/2047.5
    if os.path.exists(filename.strip('.bin') + ".txt"):
        with open(filename.strip('.bin') + ".txt",'r') as f:
            (date,channels,duration,samplerate) = f.read().splitlines()
        if channels.strip("Channels: ") == "AB":
            DATA = [DATA[ind::2] for ind in range(2)]
            TIME = [i/float(samplerate.strip('Samples per second: ')) for i in range(len(DATA[0]))]
            DataFrame(np.column_stack((DATA[0],DATA[1],TIME))).to_csv(newfile,index=False,header=("CH A (Volts)","CH B (Volts)","Time (micro-s)"))
        else:
            TIME = [i/float(samplerate.strip('Samples per second: ')) for i in range(len(DATA))] 
            DataFrame(np.column_stack((DATA,TIME))).to_csv(newfile,index=False,header=("Voltage (Volts)","Time (micro-s)"))
    else:
        print('Accompanying text file is not in the directory.')
    
def datafromcsv(filename):
    '''returns a numpy array of float values representing voltage and time.
       One column can be accessed by calling the label: data['V'] or data['t']'''
    with open(filename) as f:
        if f.readline().count(',') == 1:
            labels = ['V','t']
        elif f.readline().count(',') == 2:
            labels = ['A','B','t']
        else:
            print('File is not 2 or 3 columns')
            return ValueError
    D = np.genfromtxt(filename,  delimiter = ',',skip_header=1,names=labels)
    return D

class UniqueFilename:
    
    def __init__(self,directory,prefix,ext):
        i = 1
        date = strftime("%Y-%m-%d")
        self.path = directory
        while os.path.exists("%s%s_%s_%.f.%s" % 
                             (directory,prefix,date,i,ext)):
            i += 1
        self.filename = "{}_{}_{}.{}".format(prefix,date,i,ext)
        self.pathname = "{}{}".format(self.path,self.filename)
        
def unique_filename(directory,prefix,ext):
    '''Generates a unique, date stamped filename in the specified directory
       with the given prefix and extension'''
    i = 1
    while os.path.exists("%s%s%s_%.f.%s" % 
                         (directory,prefix,strftime("%Y-%m-%d"),i,ext)):
        i += 1
    fname = "%s%s%s_%.f.%s" % (directory,prefix,strftime("%Y-%m-%d"),i,ext)
    return fname
    
def plot_flux_tuning(dataFile,real_imag,fpoints,Istart,Istop,Ipoints,save=True):
    '''imports from flux spectroscopy csv file and plots as a 2D contour.
       real_imag MUST be exactly 'real' or 'imag', case sensitive.
       fpoints is number of points in frequency sweep. Istart/Istop is initial/final biasing current (mA).
       Ipoints is number of points in current sweep. - JF'''
    import matplotlib.pyplot as plt
    data = np.fromfile(dataFile).reshape(Ipoints,fpoints,3)
    Y = np.linspace(Istart,Istop,Ipoints)
    if real_imag == 'real':
        Z = data[:,:,0]
    elif real_imag == 'imag':
        Z = data[:,:,1]
    else:
        return ValueError
    plt.contourf(data[0,:,2],Y,Z,100)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Bias Current (mA)')
    plt.title('Transmon Frequency Flux Tuning')
    if save:
        plt.savefig(dataFile.strip('.csv')+'.png')
    plt.show()
    
def PSD(dBm,BW):
    '''compute the noise density in dBm/Hz from dBm power and bandwidth(Hz) read off spectrum analyzer'''
    return dBm - 10*np.log10(BW)
def totalNoisePower(PSD,BW):
    '''compute the total noise power in dBm, given the power spectral density in dBm/Hz and the bandwidth in Hz'''
    return PSD + 10*np.log10(BW)

def h5_tree(val, pre=''):
    '''
    given an hdf5 object, print out all groups, attributes, datasets in a heirarchal structure.

    Args:
        val (h5py.File object): the hdf5 object you want to explore.
        pre (string, optional): leave this blank when calling. It's used to make the tree structure. Defaults to ''.

    Returns:
        None.

    Example usage:
    with h5py.File(r"path/to/file",'r') as f:
        h5_tree(f)
    '''
    items = len(val)
    for key, val in val.items():
        items -= 1
        attrkeys = [k for k in val.attrs]
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                for atkey in attrkeys:
                    print(pre + '    └** ' + atkey)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                for atkey in attrkeys:
                    print(pre + '│   ├** ' + atkey)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))   
              

def connect_SRS(mode="serial", channel='COM3', ID=115200, range='range1'):
    vs = DC205(mode,channel,ID)
    vs.config.voltage_range=range
    vs.config.output = 'on'
    return vs

def turn_off_SRS(vs, volt_unit):
    """
    Sets the SRS output to 0 V and turns off the output.
    """
    if volt_unit == "mV":
        set_flux_bias_srs_in_mV(0, 5e-5)
    elif volt_unit == "V":
        set_flux_bias_srs_in_V(0,0.01)
    vs.config.output = 'off'

def disconnect_SRS(vs):
    vs.config.output = 'off'

def turn_on_SRS(vs):
    vs.config.output = 'on'

def set_flux_bias_srs_in_V(voltage, step = 0.005, lower_bound=-16, upper_bound=16): # voltage in V
    """
    Set the flux bias using the SRS.
    voltage: voltage to set the flux bias to in V
    step: step size in V
    lower_bound: lower bound of the voltage in V
    upper_bound: upper bound of the voltage in V
    """
    global vs
    if vs.config.output == 'off':
        vs.config.output = 'on'
        vs.config.voltage_range = 'range100'
    # print(f"Output status: {vs.config.output}")
    #print(f"Voltage range: {vs.config.voltage_range}")
    if voltage > upper_bound or voltage < lower_bound:
        raise ValueError('Voltage out of range')
    start_voltage = vs.setting.voltage
    print(f"Setting FFL bias to {voltage} V from {start_voltage} V")
    if voltage <= start_voltage:
        step = -step
    voltage_list = np.round(np.arange(start_voltage, voltage + step/2, step), 7)
    for value in voltage_list[1:]:
        vs.setting.voltage = value
        #print(f"Setting voltage to: {value} V")
        time.sleep(0.5)
    #print("Voltage sweep complete.")

def set_flux_bias_srs_in_mV(voltage, step = 1e-3, lower_bound=-0.6, upper_bound=0.6): # voltage in V
    """
    Set the flux bias using the SRS.
    voltage: voltage to set the flux bias to in V
    step: step size in V
    lower_bound: lower bound of the voltage in V
    upper_bound: upper bound of the voltage in V
    """
    global vs
    if vs.config.output == 'off':
        vs.config.output = 'on'
        vs.config.voltage_range = 'range10'
    # print(f"Output status: {vs.config.output}")
    #print(f"Voltage range: {vs.config.voltage_range}")
    if voltage > upper_bound or voltage < lower_bound:
        raise ValueError('Voltage out of range')
    start_voltage = vs.setting.voltage
    print(f"Setting FFL bias to {voltage*1e3} mV from {start_voltage*1e3} mV")
    if voltage <= start_voltage:
        step = -step
    voltage_list = np.round(np.arange(start_voltage, voltage + step/2, step), 7)
    for value in voltage_list[1:]:
        vs.setting.voltage = value
        #print(f"Setting voltage to: {value} V")
        time.sleep(0.5)
    #print("Voltage sweep complete.")

def create_instrument_connections(client, instrument_list):
    instruments = {}
    instrument_count = {}
    
    for instrument in instrument_list:
        name, config = instrument
        
        # Count occurrences of each instrument name
        if name not in instrument_count:
            instrument_count[name] = 1
        else:
            instrument_count[name] += 1
    
    # Reset counters for the actual connections
    seen_instruments = {}
    
    for instrument in instrument_list:
        name, config = instrument
        
        # Only append config['name'] for instruments that appear multiple times
        if instrument_count[name] > 1:
            if name not in seen_instruments:
                seen_instruments[name] = 1
            else:
                seen_instruments[name] += 1
            
            if 'name' in config and config['name']:
                unique_name = f"{name}_{config['name']}"
            else:
                unique_name = f"{name}_{seen_instruments[name]}"
        else:
            unique_name = name
            
        try:
            instruments[unique_name] = client.connectToInstrument(name, config)
            print(f"Successfully connected to {unique_name}")
        except Exception as e:
            print(f"Failed to connect to {name}: {e}")
    
    return instruments

def sumLor(f,A1,A2,A3,f1,shift,Gamma):
    return 1 - A1/(1+(2*(f-f1)/Gamma)**2) - A2/(1+(2*(f-(f1-shift))/Gamma)**2) - A3/(1+(2*(f-(f1-2*shift))/Gamma)**2)

def Lor(f,A1,f1,Gamma):
    return 1 - A1/(1+(2*(f-f1)/Gamma)**2)

def set_vna(f, span=10e6, power=5, avg=25, electrical_delay=82.584e-9):
    """
    Set the VNA to the given frequency and span, and return the center frequency of the resonance.
    f: frequency to set the VNA to in GHz
    span: span of the VNA in MHz
    power: power to set the VNA to in dBm
    avg: number of averages to take
    """
    global VNA
    VNA.setValue('Range type','Center - Span')
    VNA.setValue('Center frequency', f*1e9)
    VNA.setValue('Electrical Delay',electrical_delay)
    VNA.setValue('Span',span)
    VNA.setValue('Output enabled',True)
    VNA.setValue('Average',True)
    VNA.setValue('Wait for new trace', True)
    VNA.setValue('# of averages',avg)
    sleep(0.5)

def turn_off_clearing():
    """
    Turn off the clearing tone generator.
    """
    global Drive
    Drive.setValue('Output',False)
    sleep(0.05)

def get_vna_trace(f, span=10e6, power=5, avg=25, show_plot=False):
    """
    Get the VNA trace at the given frequency and span.
    f: frequency to get the VNA trace at in GHz
    span: span of the VNA in MHz
    power: power to set the VNA to in dBm
    avg: number of averages to take
    
    Returns:
    --------
    xBG : numpy array
        Frequency array in GHz
    zData : numpy array
        Complex S21 data
    """
    global VNA, lfVNA, FIG_PATH
    
    # Set up VNA parameters
    set_vna(f, span, power, avg)
    
    # Trigger measurement
    VNA.setValue('Trigger', True)
    
    # Get measurement data
    dF = VNA.getValue('S21')
    zData = dF['y']
    
    # Create frequency array from time array
    xBG = np.arange(dF['t0'], dF['t0'] + dF['shape'][0] * dF['dt'], dF['dt'])
    
    # Log data in Labber
    td = Labber.getTraceDict(zData, x0=xBG[0], x1=xBG[-1])
    lfVNA.addEntry({'VNA - S21': td})
        
    plt.figure(figsize=(10, 6))
    mag_db = 20 * np.log10(np.abs(zData))
    plt.plot(xBG, mag_db)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Log Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.title(f'VNA Trace at {f:.6f} GHz')
    plt.savefig(os.path.join(FIG_PATH, f'vna_trace_{f:.6f}'.replace('.', 'p') + '.png'))
    if show_plot:
        plt.show()
    else:
        plt.close()
    return xBG, zData

def fit_vna_trace(f, ph, detuning=0, span=10e6, power=5, avg=25, show_plot=False):
    """
    Find resonance by identifying the minimum of VNA magnitude data.
    
    Parameters:
    -----------
    f : float
        Frequency guess in GHz
    ph : float
        Flux bias in phi0
    span : float, optional
        Frequency span in Hz
    power : float, optional
        VNA power in dBm
    avg : int, optional
        Number of averages
    show_plot : bool, optional
        Whether to show the VNA trace plot
        
    Returns:
    --------
    f_phi : float
        Resonance frequency in GHz
    f_d : float
        Drive frequency in GHz
    """
    global VNA, FIG_PATH, SPATH
    X, zData = get_vna_trace(f, span, power, avg, show_plot)
    
    # Calculate magnitude in dB
    mag_db = 20 * np.log10(np.abs(zData))
    
    # Find the minimum of the magnitude (the dip)
    min_idx = np.argmin(mag_db)
    f_phi = X[min_idx]
    
    # For now, set f_d the same as f_phi
    f_d = f_phi - detuning
    
    logging.info(f'Resonance found at f_phi = {f_phi:.6f} GHz')
    
    # Create figure for fit visualization
    plt.figure(figsize=(10, 6))
    
    # Plot the magnitude data
    plt.plot(X, mag_db, 'b-', label='S21 Magnitude')
    
    # Plot the identified resonance and drive frequencies
    plt.axvline(x=f_phi, color='g', linestyle=':', linewidth=2, label=f'f_phi = {f_phi*1e-9:.6f} GHz')
    plt.axvline(x=f_d, color='orange', linestyle=':', linewidth=2, label=f'f_d = {f_d*1e-9:.6f} GHz')
    
    # Add a smoothed version of the data for clearer visualization
    window_size = min(25, len(mag_db) // 10)  # Adjust window size based on data length
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for centered smoothing
    
    if window_size > 3:  # Only smooth if we have enough points
        from scipy.signal import savgol_filter
        try:
            mag_smoothed = savgol_filter(mag_db, window_size, 2)
            plt.plot(X, mag_smoothed, 'r--', linewidth=2, label='Smoothed Data')
        except:
            pass  # Skip smoothing if it fails
    
    # Formatting
    plt.title(f'Flux Bias: {ph:.4f} $\Phi_0$')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    try:    
        fig_path = os.path.join(FIG_PATH, f'vna_fit_{ph:.4f}.png')
        # print(f"Saving figure to: {fig_path}")
        plt.savefig(fig_path)
    except:
        fig_path = os.path.join(FIG_PATH, f"vna_fit_{ph}.png")
        # print(f"Saving figure to: {fig_path}")
        plt.savefig(fig_path)
    
    # Save data
    s_path = os.path.join(SPATH, f'vna_fit_{ph:.4f}.pkl')
    with open(s_path, 'wb') as f:
        pickle.dump((X, zData, f_phi, f_d), f)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return f_phi, f_d
    
def turn_on_vna():
    """
    Turn on the VNA.
    """
    global VNA
    VNA.setValue('Output enabled',True)
    sleep(0.05)
    
def turn_off_vna():
    """
    Turn off the VNA and turn on the sig gen at the given frequency.
    """
    global VNA
    VNA.setValue('Output enabled',False)
    sleep(0.05)

def find_resonance(phi, span, best_fit, power=5, avg=25, electrical_delay=82.584e-9, detuning=0, show_plot=False, ofq=None, f_guess=None):
    if f_guess is None:
        f_guess = find_mapped_resonance(phi, best_fit, ofq)
        print(f"f_guess: {f_guess} GHz from the fitted Flux Curve")
    else:
        print(f"f_guess: {f_guess} GHz from the flux fit")
    turn_on_vna()
    set_vna(f_guess, span, power, avg, electrical_delay)
    f_phi, f_d = fit_vna_trace(f_guess, phi, detuning, show_plot=show_plot)
    turn_off_vna()
    return f_phi, f_d

def turn_off_LO():
    """
    Turn off the drive signal generator.
    """
    global LO
    LO.setValue('Output status',False)
    sleep(0.05)

def turn_on_LO():
    """
    Turn off the drive signal generator.
    """
    global LO
    LO.setValue('Output status',True)
    sleep(0.05)

def set_LO_tone(f, power=16):
    """
    Set the drive tone to the given frequency.
    f: frequency to set the drive tone to in Hz
    """
    global LO
    LO.setValue('Frequency',f)
    LO.setValue('Power',power)
    LO.setValue('Output status',True)
    print(f"LO tone set to {LO.getValue('Frequency')*1e-9:.6f} GHz with power {LO.getValue('Power')} dBm")
    sleep(0.05)
    
def set_clearing_tone(f, power):
    """
    Set the clearing tone to the given frequency.
    f: frequency to set the clearing tone to in GHz
    power: power to set the clearing tone to in dBm
    """
    global Drive
    Drive.setValue('Frequency',f*1e9)
    Drive.setValue('Power',power)
    Drive.setValue('Output',True)
    print(f"Clearing tone set to {Drive.getValue('Frequency')*1e-9:.6f} GHz with power {Drive.getValue('Power')} dBm")
    sleep(0.05)

def set_project(base_path, sub_dir=None):
    project_path = os.path.join(base_path, time.strftime("%m%d%y"))
    if sub_dir:
        project_path = os.path.join(project_path, sub_dir)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    print(f"Project path set to: {project_path}")
    return project_path

def write_metadata(metadata_file, acquisitionLength_sec, actualSampleRateMHz, fd, voltage, T, T_rad, ph, clearing_freq=None, clearing_power=None):
    """Write metadata to a file."""
    # Now write everything in one go
    with open(metadata_file, 'a') as f:
        # Add a separator
        f.write("\n" + "="*50 + "\n")
        
        # Write the experiment metadata
        f.write("=== Experiment Metadata ===\n")
        f.write(f"Channels: AB\n")
        f.write(f"Acquisition duration: {acquisitionLength_sec} seconds\n")
        f.write(f"Sample Rate MHz: {actualSampleRateMHz} MHz\n")
        f.write(f"Drive frequency: {fd*1e-9:.6f} GHz\n")
        f.write(f"Temperature MXC: {T} mK\n")
        f.write(f"Radiator temperature: {T_rad} mK\n")
        f.write(f"Flux bias (Phi): {ph:.6f} Phi0\n")
        f.write(f"Flux voltage: {voltage*1e3:.6f} mV\n")
        
        if clearing_freq is not None:
            f.write(f"Clearing frequency: {clearing_freq:.6f} GHz\n")
        if clearing_power is not None:
            f.write(f"Clearing power: {clearing_power:.6f} dBm\n")
        
        f.write("=== End Experiment Metadata ===\n")

def acquire_IQ_data(phi, f_clearing=None, P_clearing=None, num_traces=1, acquisitionLength_sec=5, origRateMHz=300, sampleRateMHz=10, averageTimeCycle=0, lowerBound=12, upperBound=40, spacing=2):
    """
    Acquires IQ data from the Alazar card.
    
    Parameters:
    -----------
    phi : float
        Flux point to acquire data at in phi0
    f_clearing : float
        Clearing tone frequency in GHz
    P_clearing : float
        Clearing tone power in dBm
    num_traces : int, optional
        Number of traces to acquire
    acquisitionLength_sec : float, optional
        Length of the acquisition in seconds
    origRateMHz : float, optional
        Original sample rate of the Alazar card in MHz
    sampleRateMHz : float, optional
        Sample rate to acquire data at in MHz
    averageTimeCycle : int, optional
        Number of averages to take
    lowerBound : int, optional
        Lower bound of the DA attenuator in dB
    upperBound : int, optional
        Upper bound of the DA attenuator in dB
    spacing : int, optional
        Spacing of the DA attenuator in dB
        
    Returns:
    --------
    metadata_files : list
        List of paths to all metadata files created during acquisition
    """
    global SPATH, DA, PATH_TO_EXE, device_name, LO, Drive, TWPA_PUMP
    
    metadata_files = []  # List to store all metadata file paths

    adc = ADC()
    adc.configureClock(MS_s = origRateMHz)
    adc.configureTrigger(source='EXT')
    now = perf_counter()

    # Create a safe directory name for the clearing tone parameters
    if f_clearing is not None and P_clearing is not None:
        StringForClearing = f"clearing_{f_clearing:.2f}GHz_{P_clearing:.1f}dBm".replace('.', 'p')
    elif f_clearing is not None and P_clearing is None:
        StringForClearing = f"no_clearing_REF_for_{f_clearing:.2f}GHz".replace('.', 'p')
    else:
        StringForClearing = None
        
    for ds in tqdm(np.arange(lowerBound, upperBound+1, spacing)):
        DA.setValue('Attenuation', ds)  # dB
        
        # Format the phi string properly
        phi_str = f"phi_{phi:.3f}".replace('.', 'p').strip()
        
        # Create directory structure
        if StringForClearing is not None:
            dir_path = os.path.join(SPATH, phi_str, f"DA{int(ds):02d}_SR{int(sampleRateMHz)}", StringForClearing)
        else:
            dir_path = os.path.join(SPATH, phi_str, f"DA{int(ds):02d}_SR{int(sampleRateMHz)}")
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
        except OSError as e:
            logging.error(f"Failed to create directory {dir_path}: {e}")
            raise
        
        # Create timestamp and filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        savefile = os.path.join(dir_path, f"{device_name}_{timestamp}.bin")
        metadata_file = savefile[0:-4] + ".txt"
        metadata_files.append(metadata_file)  # Add metadata file to list

        # Calculate actual sample rate
        samplesPerPoint = int(max(origRateMHz / sampleRateMHz, 1))
        actualSampleRateMHz = origRateMHz / samplesPerPoint
        
        # Run the acquisition
        logging.info(f"Running acquisition with {samplesPerPoint} samples per point")
        try:
            Creturn = subprocess.getoutput(f'"{PATH_TO_EXE}" {int(acquisitionLength_sec)} {samplesPerPoint} "{savefile}"')
            logging.info(f"Acquisition result: {Creturn}")
        except Exception as e:
            logging.error(f"Error during acquisition: {e}")
        
        # Write basic info to the metadata file
        with open(metadata_file, 'w') as f:
            f.write("=== Basic Info ===\n")
            f.write(f"Timestamp: {time.strftime('%c')}\n")
            f.write(f"Digital Attenuator: {DA.getValue('Attenuation')} dB\n")
            f.write(f"Sample rate: {actualSampleRateMHz} MHz\n")
            f.write(f"Acquisition length: {acquisitionLength_sec} seconds\n")
            f.write(f"Phi: {phi:.6f}\n")
            f.write(f"LO_frequency: {LO.getValue('Frequency')*1e-9:.6f} GHz\n")
            try:
                f.write(f"Clearing frequency: {f_clearing:.6f} GHz\n")
                f.write(f"Clearing power: {P_clearing:.6f} dBm\n")
            except:
                pass
            f.write("=== End Basic Info ===\n")
        
        sleep(0.05)

    timeCycle = perf_counter() - now
    logging.info(f'Acquisition completed in {timeCycle:.6f} seconds.')
    
    return metadata_files

def set_TWPA_pump(f=6.04, power=27):
    """
    f: frequency to set the TWPA pump to in GHz
    power: power to set the TWPA pump to in dBm
    """
    global TWPA_PUMP
    TWPA_PUMP.setValue('Frequency', f*1e9)
    TWPA_PUMP.setValue('Power', power)
    TWPA_PUMP.setValue('Output status', True)

def turn_off_TWPA_pump():
    global TWPA_PUMP
    TWPA_PUMP.setValue('Output status', False)

def turn_off_Drive():
    global Drive
    Drive.setValue('Output status', False)

# Add after imports
def verify_environment():
    """Verify all required components are in place"""
    required_files = [
        'utils.py',
        'flux_fit.py',
        'data_transfer_daemon.py',
        '.env'
    ]
    
    required_paths = [
        PATH_TO_EXE,
        drive_path
    ]
    
    # Check files
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file {file} not found")
    
    # Check paths
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path {path} not found")
    
    # Check environment variables
    try:
        load_dotenv()
        required_env = ['HPC_USERNAME', 'HPC_HOSTNAME', 'HPC_BASE_PATH']
        for env in required_env:
            if not os.getenv(env):
                raise ValueError(f"Environment variable {env} not set")
    except Exception as e:
        raise Exception(f"Environment setup failed: {str(e)}")

def pre_experiment_checks():
    """Run pre-experiment checks"""
    checks = {
        "Storage Space": {
            "check": lambda: shutil.disk_usage(SPATH).free > 50 * 1024**3,  # 50GB
            "message": "Insufficient storage space (need >50GB)"
        },
        "Instrument Connections": {
            "check": lambda: all([VNA, DA, SMU, LO, Drive, vs, TWPA_PUMP]),
            "message": "Not all instruments are connected"
        },
        "HPC SSH": {
            "check": lambda: subprocess.run(
                f"ssh -q {os.getenv('HPC_USERNAME')}@{os.getenv('HPC_HOSTNAME')} exit",
                shell=True
            ).returncode == 0,
            "message": "Cannot connect to HPC (check SSH setup)"
        }
    }
    
    failed_checks = []
    for name, check in checks.items():
        try:
            if not check["check"]():
                failed_checks.append(f"{name}: {check['message']}")
        except Exception as e:
            failed_checks.append(f"{name}: Error during check - {str(e)}")
    
    if failed_checks:
        raise RuntimeError("Pre-experiment checks failed:\n" + "\n".join(failed_checks))
    
    print("All pre-experiment checks passed!")

def start_transfer_daemon():
    """Start the data transfer daemon in a separate process"""
    import subprocess
    import sys
    
    daemon_process = subprocess.Popen(
        [sys.executable, 'data_transfer_daemon.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit and check if process is still running
    time.sleep(2)
    if daemon_process.poll() is not None:
        out, err = daemon_process.communicate()
        raise RuntimeError(f"Data transfer daemon failed to start:\n{err.decode()}")
    
    return daemon_process

def stop_transfer_daemon(daemon_process):
    """Stop the data transfer daemon"""
    daemon_process.terminate()
    try:
        daemon_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        daemon_process.kill()

def check_storage_space(path, required_gb=50):
    """Check if there's enough storage space"""
    free_space = shutil.disk_usage(path).free / (1024**3)  # Convert to GB
    if free_space < required_gb:
        raise RuntimeError(f"Insufficient storage space. Need {required_gb}GB, have {free_space:.1f}GB")
    logging.info(f"Storage space check passed: {free_space:.1f}GB available")

def verify_hpc_connection():
    """Verify HPC connection is working"""
    hpc_user = os.getenv('HPC_USERNAME')
    hpc_host = os.getenv('HPC_HOSTNAME')
    if not all([hpc_user, hpc_host]):
        raise ValueError("HPC credentials not found in environment")
    
    result = subprocess.run(
        f"ssh -q {hpc_user}@{hpc_host} echo 'Connection test'",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"HPC connection failed: {result.stderr}")
    logging.info("HPC connection verified successfully")

    
def adjust_range(f, s21, f_min, f_max):
    mask = (f >= f_min) & (f <= f_max)
    return f[mask], s21[mask]

def process_powersweep_fits(lf, nEntries, power, pdf_name, fmin, fmax):
    """
    Process fits for the given Labber log file and save the results to a PDF.
    
    Parameters:
    lf (Labber.LogFile): The Labber log file object.
    nEntries (int): The number of entries in the log file.
    power (list): The list of power values for each entry.
    pdf_name (str): The name of the PDF file to save the plots.
    fmin (float): The minimum frequency for the fit range.
    fmax (float): The maximum frequency for the fit range.
    
    """
    from matplotlib.backends.backend_pdf import PdfPages
    fits = {'f': [], 'Qint': [], 'Qext': [], 'power': [], 'Qtot': [], 'f_error': []}
    fit_objects = []

    # Create a PdfPages object to save all plots in one PDF
    with PdfPages(pdf_name) as pdf:
        for n in range(nEntries):
            (frequency, S21) = lf.getTraceXY(entry=n)
            frequency, S21 = adjust_range(frequency, S21, fmin, fmax)
            r = reflection.LinearReflectionFitter(frequency, S21)
            fit_objects.append(r)
            # Generate plots
            fig, (ax_magnitude, ax_phase, ax_complex) = see.triptych(
                resonator=r, plot_initial=False, frequency_scale=1e-9,
                figure_settings={'figsize': (16, 8), 'dpi': 300}
            )
            # Append power to the plot title
            fig.suptitle(f'DA Attenuation: {power[n]} dBm')
            ax_complex.legend()
            # Save the current figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)
            # Append the fit results to the dictionary
            fits['f'].append(r.resonance_frequency)
            fits['Qtot'].append(r.Q_t)
            fits['Qint'].append(r.Q_i)
            fits['Qext'].append(r.Q_c)
            fits['power'].append(power[n])
            fits['f_error'].append(r.f_r_error)

    return fits, fit_objects

def process_fluxsweep_fits(lf, nEntries, voltage, pdf_name, span):
    from matplotlib.backends.backend_pdf import PdfPages
    fits = {'f': [], 'Qint': [], 'Qext': [], 'voltage': [], 'Qtot': [], 'f_error': []}
    fit_objects = []

    # Create a PdfPages object to save all plots in one PDF
    with PdfPages(pdf_name) as pdf:
        for n in range(nEntries):
            (frequency, S21) = lf.getTraceXY(entry=n)
            # Calculate the log magnitude of S21
            S21_log_mag = 20 * np.log10(np.abs(S21))
            # Find the index of the dip in the log magnitude
            dip_index = np.argmin(S21_log_mag)
            # Determine the center frequency
            center_frequency = frequency[dip_index]
            # Set fmin and fmax based on the span
            fmin = center_frequency - span / 2
            fmax = center_frequency + span / 2
            # Adjust the frequency and S21 range
            frequency, S21 = adjust_range(frequency, S21, fmin, fmax)
            r = reflection.LinearReflectionFitter(frequency, S21)
            fit_objects.append(r)
            # Generate plots
            fig, (ax_magnitude, ax_phase, ax_complex) = see.triptych(
                resonator=r, plot_initial=False, frequency_scale=1e-9,
                figure_settings={'figsize': (16, 8), 'dpi': 300}
            )
            # Append voltage to the plot title
            fig.suptitle(f'Voltage: {voltage[n]} mV')
            ax_complex.legend()
            # Save the current figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)
            # Append the fit results to the dictionary
            fits['f'].append(r.resonance_frequency)
            fits['Qtot'].append(r.Q_t)
            fits['Qint'].append(r.Q_i)
            fits['Qext'].append(r.Q_c)
            fits['voltage'].append(voltage[n])
            fits['f_error'].append(r.f_r_error)

    return fits, fit_objects

def plot_gain_improvement_heatmap(df, s21_ref, improvement_threshold=None, figsize=(12, 8)):
    """
    Plot SNR improvement as a function of pump frequency and power.
    
    Parameters:
    df (DataFrame): DataFrame containing pump_freq, pump_power, and S21 data
    s21_ref (array): Reference S21 data (when pump is off)
    improvement_threshold (float, optional): If provided, only show improvements above this threshold
    figsize (tuple): Figure size (width, height)
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """    
    # Extract unique values to use for the plot
    unique_pump_freqs = df['pump_freq'].unique()
    unique_pump_powers = df['pump_power'].unique()
    
    # Create a 2D array to hold the SNR improvement values
    snr_improvement_2d = np.full((len(unique_pump_powers), len(unique_pump_freqs)), np.nan)
    
    # Fill in the values from the DataFrame
    for i, power in enumerate(unique_pump_powers):
        for j, freq in enumerate(unique_pump_freqs):
            # Get the row that matches this power and frequency combination
            mask = (df['pump_freq'] == freq) & (df['pump_power'] == power)
            if mask.any():  # Check if this combination exists in the DataFrame
                # Get the SNR improvement value for this combination
                snr_value = df.loc[mask, 'snr_improvement'].values[0]
                
                # If snr_value is a 1D array, compute its mean
                if isinstance(snr_value, np.ndarray) and snr_value.ndim > 0:
                    snr_value = np.mean(snr_value)
                    
                # Apply threshold if specified
                if improvement_threshold is not None and snr_value < improvement_threshold:
                    snr_improvement_2d[i, j] = np.nan
                else:
                    snr_improvement_2d[i, j] = snr_value
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find min and max values for colorbar, excluding NaNs
    vmin = np.nanmin(snr_improvement_2d)
    vmax = np.nanmax(snr_improvement_2d)
    
    # Check if we have valid data
    if np.isnan(vmin) or np.isnan(vmax):
        print("No valid data to plot after applying threshold.")
        return fig
    
    # Set up the colormap to handle NaN values with a distinct color (gray)
    cmap = plt.cm.viridis
    cmap.set_bad(color='lightgray')
    
    # Check if we need a diverging colormap
    has_negative = vmin < 0
    has_positive = vmax > 0
    
    if has_negative and has_positive:
        # Use TwoSlopeNorm for diverging data
        try:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        except ValueError:
            # Handle edge cases where vmin and vmax are too close
            if abs(vmax - vmin) < 1e-10:
                vmin -= 0.1
                vmax += 0.1
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        # Use standard normalization for single-sided data
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    c = ax.pcolormesh(unique_pump_freqs, unique_pump_powers, snr_improvement_2d, 
                      norm=norm, cmap=cmap, shading='auto')
    
    # Add labels and title
    title = 'SNR Improvement as a Function of Pump Power and Frequency'
    if improvement_threshold is not None:
        title += f' (Threshold: {improvement_threshold} dB)'
    ax.set_xlabel('Pump Frequency (GHz)', fontsize=12)
    ax.set_ylabel('Pump Power (dBm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('Gain (dB)', fontsize=12)
    
    # Find the best combination if we have valid data
    if not np.all(np.isnan(snr_improvement_2d)):
        best_i, best_j = np.unravel_index(np.nanargmax(snr_improvement_2d), snr_improvement_2d.shape)
        best_power = unique_pump_powers[best_i]
        best_freq = unique_pump_freqs[best_j]
        best_snr = snr_improvement_2d[best_i, best_j]
        
        # Mark the best combination
        ax.plot(best_freq, best_power, 'o', color='lime', markersize=10, markeredgecolor='black')
        ax.text(best_freq, best_power*0.99, f"Best: {best_snr:.2f} dB", 
                color='black', fontweight='bold', ha='center', va='top')
        
        print(f"Best Gain improvement of {best_snr:.2f} dB at pump frequency = {best_freq:.4f} GHz and pump power = {best_power:.2f} dBm")
    
    # Add a legend entry for missing or filtered data
    import matplotlib.patches as mpatches
    missing_label = "Missing data"
    if improvement_threshold is not None:
        missing_label += f" or < {improvement_threshold} dB"
    missing_patch = mpatches.Patch(color='lightgray', label=missing_label)
    ax.legend(handles=[missing_patch], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    return fig