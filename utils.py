import logging
import os
import pickle
import struct
import time
from time import sleep, strftime

import fitTools.quasiparticleFunctions as qp
import h5py
import Labber
import matplotlib.pyplot as plt
import numpy as np
from fitTools.utilities import Watt2dBm, dBm2Watt
from pandas import DataFrame
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

def initialize_instruments(vna, da=None, smu=None, lo=None, drive=None, srs=None):
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
    global VNA, DA, SMU, LO, Drive, vs
    VNA = vna
    DA = da
    SMU = smu
    LO = lo
    Drive = drive
    vs = srs
    
def initializeLabberlogging(lfvna):
    global lfVNA
    lfVNA = lfvna

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

def disconnect_SRS(vs):
    vs.config.output = 'off'

def turn_on_SRS(vs):
    vs.config.output = 'on'

def set_flux_bias_srs(voltage, step = 1e-3, lower_bound=-0.6, upper_bound=0.6): # voltage in V
    """
    Set the flux bias using the SRS电源.
    voltage: voltage to set the flux bias to in V
    step: step size in V
    lower_bound: lower bound of the voltage in V
    upper_bound: upper bound of the voltage in V
    """
    global vs
    if vs.config.output == 'off':
        vs.config.output = 'on'
        vs.config.voltage_range = 'range10'
    print(f"Output status: {vs.config.output}")
    print(f"Voltage range: {vs.config.voltage_range}")
    if voltage > upper_bound or voltage < lower_bound:
        raise ValueError('Voltage out of range')
    print(f"Setting FFL bias to {voltage*1e3} mV")
    start_voltage = vs.setting.voltage
    print(f"Starting voltage: {start_voltage} V")
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
    VNA.setValue('Span',span*1e6)
    VNA.setValue('Output enabled',True)
    VNA.setValue('Output power',power)
    VNA.setValue('# of averages',avg)
    sleep(1)


def get_vna_trace(f, span=10e6, power=5, avg=25, show_plot=False):
    """
    Get the VNA trace at the given frequency and span.
    f: frequency to get the VNA trace at in GHz
    span: span of the VNA in MHz
    power: power to set the VNA to in dBm
    avg: number of averages to take
    """
    global VNA, lfVNA
    set_vna(f, span, power, avg)
    VNA.setValue('Trigger',True)
    dF = VNA.getValue('S21')
    zData = dF['y']
    xBG = np.arange(dF['t0'],dF['t0']+dF['shape'][0]*dF['dt'],dF['dt'])
    td = Labber.getTraceDict(zData,x0=xBG[0],x1=xBG[-1])
    lfVNA.addEntry({'VNA - S21':td})
    if show_plot:
        plt.figure()
        plt.plot(zData)
        plt.show()
    return dF

def fit_vna_trace(f, ph, span=10e6, power=5, avg=25, show_plot=False):
    """
    Fit VNA trace around a given frequency guess to find exact resonance.
    
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
        
    Returns:
    --------
    f_phi : float
        Resonance frequency in GHz
    f_d : float
        Drive frequency in GHz
    """
    global VNA
    X, zData = get_vna_trace(f, span, power, avg, show_plot)
    
    # Find index of frequency closest to the guess frequency
    fguess_idx = np.abs(X - f).argmin()
    
    plt.figure()
    try:
        logging.info('Trying to fit a double resonance...')
        pars,cov = curve_fit(sumLor,X,zData.real,p0 = [0.8,0.2,0.5,X[fguess_idx],0,0.00025],bounds=([0.2,0,0.2,min(X),-0.005,0.00005],[2,0.5,1,max(X),0.005,0.0005]))
        plt.plot(X,zData.real,label='data')
        plt.plot(X,sumLor(X,*pars),label='fit')
        plt.plot(X,Lor(X,pars[0],pars[3],pars[5]),label='Lor1')
        plt.plot(X,Lor(X,pars[1],pars[3]+pars[4],pars[5]),label='Lor2')
        A1, A2, A3, f1, shift, Gamma = pars
        logging.info('Fit params: A1 = %.2f, A2 = %.2f, A3 = %.2f, f1 = %.6f, shift = %.6f, Gamma = %.6f'%(A1,A2,A3,f1,shift,Gamma))
        
        if abs(shift) < 3e-4:
            f_phi = f1
            fd = f_phi
        else:
            f_phi = f1
            fd = f1 + shift
            if fd > f_phi:
                f_phi, fd = fd, f_phi
    except:
        logging.warning('Trying to fit a single resonance...')
        try:
            pars,cov = curve_fit(Lor,X,zData.real,p0 = [0.8,X[fguess_idx],0.00025],bounds=([0.2,5.78,0.00005],[2,5.78,0.0005]))
            plt.plot(X,zData.real,label='data')
            plt.plot(X,Lor(X,*pars),label='fit')
            A1, f1, Gamma = pars
            logging.info('Fit params: A1 = %.2f, f1 = %.6f, Gamma = %.6f'%(A1,f1,Gamma))
            f_phi = f1
            fd = f_phi
        except:
            logging.warning('Unable to fit resonance. Using the minimum of real part as estimated resonance.')
            plt.plot(X,zData.real,label='data')
            pars = np.array([1.5,X[fguess_idx],0.00025])
            cov = np.ones((len(pars),len(pars)))*np.infty
            plt.plot(X,Lor(X,*pars),label='unfitted guess')
            A1, f1, Gamma = pars
            f_phi = f1
            fd = f_phi
    
    plt.title('Flux Bias: %.4f $\Phi_0$'%ph)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Re(S21)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('figures/vna_fit_%.4f.png'%ph)
    
    with open('data/vna_fit_%.4f.pkl'%ph, 'wb') as f:
        pickle.dump((X, zData, pars, cov), f)
    
    plt.close()
    
    return f_phi, fd
    
def turn_on_vna():
    """
    Turn on the VNA.
    """
    global VNA
    VNA.setValue('Output enabled',True)
    
def turn_off_vna():
    """
    Turn off the VNA and turn on the sig gen at the given frequency.
    """
    global VNA
    VNA.setValue('Output enabled',False)

def find_resonance(phi, span, best_fit, power=5, avg=25, electrical_delay=82.584e-9, show_plot=False):
    f_guess = find_mapped_resonance(phi, best_fit)
    print(f"f_guess: {f_guess} GHz")
    set_vna(f_guess, span, power, avg, electrical_delay)
    f_phi, fd = fit_vna_trace(f_guess, phi, show_plot=show_plot)
    turn_off_vna()
    return f_phi, fd


def set_drive_tone(f, power=16):
    """
    Set the drive tone to the given frequency.
    f: frequency to set the drive tone to in GHz
    """
    global Drive
    Drive.setValue('Frequency',f*1e9)
    Drive.setValue('Power',power)
    Drive.setValue('Output status',True)

def set_clearing_tone(f, power):
    """
    Set the clearing tone to the given frequency.
    f: frequency to set the clearing tone to in GHz
    power: power to set the clearing tone to in dBm
    """
    global LO
    LO.setValue('Frequency',f*1e9)
    LO.setValue('Power',power)
    LO.setValue('Output status',True)

def set_project(base_path, sub_dir=None):
    project_path = os.path.join(base_path, time.strftime("%m%d%y"))
    if sub_dir:
        project_path = os.path.join(project_path, sub_dir)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    print(f"Project path set to: {project_path}")
    return project_path

def write_metadata(savefile, acquisitionLength_sec, actualSampleRateMHz, fd, voltage, T, T_rad, ph, clearing_freq=None, clearing_power=None):
    """Write metadata to a file."""
    with open(savefile[0:-4] + ".txt", 'a') as f:
        f.write("Channels: " + 'AB' + '\n')
        f.write("Acquisition duration: " + str(acquisitionLength_sec) + " seconds." + '\n')
        f.write("Sample Rate MHz: " + str(actualSampleRateMHz) + '\n')
        f.write("LO frequency: " + str(fd) + " GHz\n")
        f.write("flux bias: " + str(V) + " mV\n")
        f.write("Temperature: " + str(T) + ' mK\n')
        f.write("Radiator temperature: " + str(T_rad) + ' mK\n')
        f.write("PHI: " + str(ph) + '\n')
        f.write("Voltage: " + str(voltage) + " mV\n")
        f.write("Clearing freq:" + str(clearing_freq) + " GHz\n")
        f.write("Clearing power:" + str(clearing_power) + " dBm\n")

def acquire_IQ_data(phi, num_traces=1, acquisitionLength_sec=5, origRateMHz=300, sampleRateMHz=10, averageTimeCycle=0, lowerBound=12, upperBound=40):
    """
    Acquires IQ data from the Alazar card.
    phi: flux point to acquire data at
    f_drive: drive tone frequency in GHz
    num_traces: number of traces to acquire
    acquisitionLength_sec: length of the acquisition in seconds
    origRateMHz: original sample rate of the Alazar card in MHz
    sampleRateMHz: sample rate to acquire data at in MHz
    T: temperature of the resonator in mK
    averageTimeCycle: number of averages to take
    lowerBound: lower bound of the DA attenuator in dB
    upperBound: upper bound of the DA attenuator in dB
    """
    global SPATH, DA, PATH_TO_EXE, device_name

    adc = ADC()
    adc.configureClock(MS_s = origRateMHz)
    adc.configureTrigger(source='EXT')
    now = perf_counter()

    for ds in tqdm(np.arange(lowerBound, upperBound+1, 2)):
        DA.setValue('Attenuation', ds)# dB
        StringForFlux = r'{:3.0f}flux\DA{:2.0f}_SR{:d}MHz'.format(phi * 1000, ds, sampleRateMHz)
        path = os.path.join(SPATH, "{}".format(StringForFlux))
        
        os.makedirs(path, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        savefile = os.path.join(path, '{}_{}.bin'.format(device_name, timestamp))
        samplesPerPoint = int(max(origRateMHz / sampleRateMHz, 1))
        actualSampleRateMHz = origRateMHz / samplesPerPoint
        
        Creturn = subprocess.getoutput('"{}" {} {} "{}"'.format(PATH_TO_EXE, int(acquisitionLength_sec), samplesPerPoint, savefile))
        logging.info(Creturn)
        
        with open(savefile[0:-4] + ".txt", 'w') as f:
            f.write(time.strftime("%c") + '\n')
            f.write(f"DA power: {DA.getValue('Attenuation')} dB\n")

        sleep(0.05)

    timeCycle = perf_counter() - now
    logging.info(f'This step took {timeCycle:.6f} seconds.')
    return savefile