import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def voltage_to_flux_quanta(voltage, best_params):
    """
    Converts voltage to normalized flux quanta using the fitted parameters.

    Parameters:
    -----------
    voltage : float or array-like
        The voltage value(s).
    best_params : dict
        The dictionary containing the best-fit parameters from fit_resonance_data.

    Returns:
    --------
    flux_quanta : float or array-like
        The normalized flux quanta value(s).
    """
    if best_params is None:
        raise ValueError("Fit parameters are not available. Run fit_resonance_data first.")

    V_offset = best_params['V_offset']
    V_period = best_params['V_period']

    flux_quanta = (voltage - V_offset) / V_period
    return flux_quanta

def flux_quanta_to_frequency(flux_quanta, best_params, ofq=None):
    """
    Predicts the resonant frequency from normalized flux quanta using the fitted parameters.

    Parameters:
    -----------
    flux_quanta : float or array-like
        The normalized flux quanta value(s).
    best_params : dict
        The dictionary containing the best-fit parameters from fit_resonance_data.

    Returns:
    --------
    frequency : float or array-like
        The predicted resonant frequency value(s).
    """
    if best_params is None:
        raise ValueError("Fit parameters are not available. Run fit_resonance_data first.")

    w0 = best_params['w0']
    q0 = best_params['q0']
    V_period = best_params['V_period'] #these are not used, but kept for consistency
    V_offset = best_params['V_offset'] #these are not used, but kept for consistency

    sin_term = np.sin(np.pi * flux_quanta / 2)
    clipped_sin = np.clip(sin_term, -0.99, 0.99)
    arctanh_term = np.arctanh(clipped_sin)
    numerator = clipped_sin * arctanh_term
    denominator = 1 - clipped_sin * arctanh_term
    safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    exponent = -0.5
    frequency = w0 * (1 + q0 * numerator / safe_denominator) ** exponent

    # If ofq is provided, handle the high flux regions
    if ofq is not None:
        # Handle positive branch
        pos_mask = flux_quanta > 0.46
        if np.any(pos_mask):
            for idx in np.where(pos_mask)[0]:
                fq = flux_quanta[idx]
                flux_quanta_closest = ofq['positive_branch']['flux'][
                    np.argmin(np.abs(ofq['positive_branch']['flux'] - fq))
                ]
                frequency[idx] = ofq['positive_branch']['resonances'][
                    np.argmin(np.abs(ofq['positive_branch']['flux'] - flux_quanta_closest))
                ]
                print(f"flux_quanta[{idx}] = {fq:.3f} > 0.46. Using measured data value instead of fitted data.")

        # Handle negative branch
        neg_mask = flux_quanta < -0.46
        if np.any(neg_mask):
            for idx in np.where(neg_mask)[0]:
                fq = flux_quanta[idx]
                flux_quanta_closest = ofq['negative_branch']['flux'][
                    np.argmin(np.abs(ofq['negative_branch']['flux'] - fq))
                ]
                frequency[idx] = ofq['negative_branch']['resonances'][
                    np.argmin(np.abs(ofq['negative_branch']['flux'] - flux_quanta_closest))
                ]
                print(f"flux_quanta[{idx}] = {fq:.3f} < -0.46. Using measured data value instead of fitted data.")

    return frequency

def phi_to_voltage(phi, best_params, unit='mV'):
    """
    Converts normalized flux quanta (phi) to voltage using the fitted parameters.

    Parameters:
    -----------
    phi : float or array-like
        The normalized flux quanta value(s).
    best_params : dict
        The dictionary containing the best-fit parameters from fit_resonance_data.

    Returns:
    --------
    voltage : float or array-like
        The voltage value(s).
    """
    if best_params is None:
        raise ValueError("Fit parameters are not available. Run fit_resonance_data first.")

    V_offset = best_params['V_offset']
    V_period = best_params['V_period']

    voltage = phi * V_period + V_offset
    if unit == 'V':
        voltage = voltage * 1e-3
    elif unit == 'mV':
        voltage = voltage
    else:
        raise ValueError("Invalid unit. Please use 'mV' or 'V'.")
    return voltage

def from_flux_fit(phi_arr, best_fit, ofq=None):
    v_arr = phi_to_voltage(phi_arr, best_fit)
    f_arr = flux_quanta_to_frequency(phi_arr, best_fit, ofq)
    return v_arr, f_arr

def find_mapped_resonance(phi, best_fit, ofq=None):
    """
    Finds the mapped resonance frequency for a given flux point.
    
    Parameters:
    -----------
    phi : float
        The flux point to find the mapped resonance for.
    best_fit : dict
        The best-fit parameters from fit_resonance_data.
        
    Returns:
    --------
    f_arr : float
        The mapped resonance frequency in GHz.
    """
    f_arr = flux_quanta_to_frequency(phi, best_fit, ofq)
    return f_arr

def plot_sweep_points(v_arr, f_arr, phi_arr):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(v_arr, f_arr, ".")
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Frequency (GHz)")
    plt.title("Flux Points to be swept")

    plt.subplot(1, 2, 2)
    plt.plot(phi_arr, f_arr, ".")
    plt.xlabel("Phi")
    plt.ylabel("Frequency (GHz)")
    plt.title("Phi Points to be swept")

    plt.tight_layout()
    plt.show()



# Define the correct flux-dependent resonant frequency formula from the paper
def resonant_frequency(bias_voltage, w0, q0, V_period, V_offset):
    """
    Implements the exact resonant frequency formula for a SQUID from the paper:
    
    ω(φ) = ω0 * [1 + q0 * (sin(πφ/2)tanh^(-1)(sin(πφ/2))) / (1 - sin(πφ/2)tanh^(-1)(sin(πφ/2)))]^(-1/2)
    
    Parameters:
    -----------
    bias_voltage : array-like
        The bias voltage values
    w0 : float
        Base resonant frequency
    q0 : float
        Josephson inductance participation ratio at zero flux
    V_period : float
        Voltage period corresponding to one flux quantum
    V_offset : float
        Voltage offset to align the flux minimum
    
    Returns:
    --------
    frequency : array-like
        Resonant frequency values
    """
    # Convert voltage to normalized flux (Φ/Φ0)
    phi = (bias_voltage - V_offset) / V_period
    
    # Compute the formula terms
    sin_term = np.sin(np.pi * phi / 2)
    
    # To avoid numerical issues, clip the sin values
    clipped_sin = np.clip(sin_term, -0.99, 0.99)
    
    # Calculate arctanh (inverse hyperbolic tangent)
    arctanh_term = np.arctanh(clipped_sin)
    
    # Calculate numerator and denominator
    numerator = clipped_sin * arctanh_term
    denominator = 1 - clipped_sin * arctanh_term
    
    # Handle potential numerical issues in denominator
    safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    
    # Calculate the frequency
    exponent = -0.5
    frequency = w0 * (1 + q0 * numerator / safe_denominator) ** exponent
    
    return frequency


def get_voltage_array(V_start, V_end, steps):
    # Create voltage array corresponding to the resonance data
    voltages = np.arange(V_start, V_end+steps, steps)
    return voltages

def other_flux_quanta_resonances(resonances, voltages, remove_first=0, remove_last=0, best_fit=None):
    # Filter the data if requested
    if remove_first > 0 or remove_last > 0:
        # Create a mask for keeping data
        mask = np.zeros(len(resonances), dtype=bool) 
        if remove_first > 0:
            mask[:remove_first] = True
        if remove_last > 0:
            mask[-remove_last:] = True
            
        filtered_resonances = resonances[mask]
        filtered_voltages = voltages[mask]
        filtered_flux_quanta = voltage_to_flux_quanta(filtered_voltages, best_fit)

    # Prepare data for JSON output
    data = {
        "negative_branch": {
            "flux": filtered_flux_quanta[:remove_first],
            "voltage": filtered_voltages[:remove_first],
            "resonances": filtered_resonances[:remove_first]
        },
        "positive_branch": {
            "flux": filtered_flux_quanta[-remove_last:],
            "voltage": filtered_voltages[-remove_last:],
            "resonances": filtered_resonances[-remove_last:]
        }
    }

    return data

# Filter the data if needed
def fit_resonance_data(resonances, V_start, V_end, steps, remove_first=3, remove_last=2):
    """
    Fit the resonance data to the flux-dependent formula.
    Optionally removes outlier points.
    """
    # Create voltage array corresponding to the resonance data
    voltages = np.arange(V_start, V_end+steps, steps)
    

    # Check if the length of voltages matches the length of resonances
    if len(voltages) != len(resonances):
        raise ValueError("The length of voltages and resonances must be the same.")
    
    # Filter the data if requested
    if remove_first > 0 or remove_last > 0:
        # Create a mask for keeping data
        mask = np.ones(len(resonances), dtype=bool)
        if remove_first > 0:
            mask[:remove_first] = False
        if remove_last > 0:
            mask[-remove_last:] = False
            
        filtered_resonances = resonances[mask]
        filtered_voltages = voltages[mask]
        
        print(f"Original data points: {len(resonances)}")
        print(f"Filtered data points: {len(filtered_resonances)}")
        print(f"Removed points at voltages: {voltages[~mask]}")
    else:
        filtered_resonances = resonances
        filtered_voltages = voltages
    
    # Initial parameter guesses based on the data and paper
    w0_guess = np.max(filtered_resonances)  # Max frequency as baseline
    q0_guess = 0.01  # Small participation ratio similar to paper's 0.01341
    V_period_guess = (V_end - V_start) / 2  # Assuming one period in the full range
    V_offset_guess = (V_start + V_end) / 2  # Center of the range
    
    # Try different initialization parameters
    best_params = None
    best_error = float('inf')
    
    # Grid search over V_period and V_offset
    for V_period_try in np.linspace(V_period_guess/2, V_period_guess*2, 10):
        for V_offset_try in np.linspace(min(voltages), max(voltages), 15):
            try:
                # Create a wrapper function with fixed V_period and V_offset
                def fit_func(x, w0, q0):
                    return resonant_frequency(x, w0, q0, V_period_try, V_offset_try)
                
                # Perform the fit
                popt, pcov = curve_fit(
                    fit_func, 
                    filtered_voltages, 
                    filtered_resonances,
                    p0=[w0_guess, q0_guess],
                    bounds=([5.7, 0.0001], [5.9, 0.1]),
                    maxfev=10000
                )
                
                # Calculate the error
                predicted = fit_func(filtered_voltages, *popt)
                error = np.sum((predicted - filtered_resonances)**2)
                
                # Update best parameters if this fit is better
                if error < best_error:
                    best_error = error
                    best_params = {
                        'w0': popt[0],
                        'q0': popt[1],
                        'V_period': V_period_try,
                        'V_offset': V_offset_try,
                        'error': error
                    }
            except Exception as e:
                # Skip failed fits
                continue
    
    if best_params is None:
        print("Failed to find a good fit.")
        return None
    
    # Print the best fitting parameters
    print("\nBest fitting parameters:")
    print(f"w0 = {best_params['w0']:.6f} GHz")
    print(f"q0 = {best_params['q0']:.6f}")
    print(f"V_period = {best_params['V_period']:.6f}")
    print(f"V_offset = {best_params['V_offset']:.6f}")
    print(f"Sum of squared error = {best_params['error']:.6e}")
    
    # Create a plot of the data and fit
    plt.figure(figsize=(12, 8))
    
    # Plot the original data
    plt.scatter(voltages, resonances, color='blue', label='Original Data')
    
    # Highlight removed points if any
    if remove_first > 0 or remove_last > 0:
        plt.scatter(
            voltages[~mask], 
            resonances[~mask], 
            color='red', 
            marker='x', 
            s=100, 
            label='Removed Points'
        )
    
    # Create a dense array for plotting the fit
    x_dense = np.linspace(min(voltages), max(voltages), 1000)
    y_fit = resonant_frequency(
        x_dense, 
        best_params['w0'], 
        best_params['q0'], 
        best_params['V_period'], 
        best_params['V_offset']
    )
    
    # Plot the fit
    plt.plot(
        x_dense, 
        y_fit, 
        'r-', 
        lw=2, 
        label=f"Fit: w0={best_params['w0']:.4f}, q0={best_params['q0']:.6f}"
    )
    
    # Plot the fit at the filtered data points
    y_fit_at_points = resonant_frequency(
        filtered_voltages, 
        best_params['w0'], 
        best_params['q0'], 
        best_params['V_period'], 
        best_params['V_offset']
    )
    
    plt.scatter(
        filtered_voltages, 
        y_fit_at_points, 
        color='green', 
        marker='+', 
        s=100, 
        label='Fit at data points'
    )
    
    # Calculate and display RMSE
    rmse = np.sqrt(np.mean((filtered_resonances - y_fit_at_points)**2))
    
    # Create a secondary x-axis for normalized flux
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    # Set the flux range based on the best parameters
    flux_min = (min(voltages) - best_params['V_offset']) / best_params['V_period']
    flux_max = (max(voltages) - best_params['V_offset']) / best_params['V_period']
    ax2.set_xlim(flux_min, flux_max)
    ax2.set_xlabel('Normalized Flux (Φ/Φ₀)')
    
    # Add labels and title
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Resonance Frequency (GHz)')
    ax1.set_title(f"Flux-Dependent Resonant Frequency (RMSE: {rmse:.6f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # crop y-axis to the range of the data
    ax1.set_ylim(min(filtered_resonances)-0.01, max(filtered_resonances)+0.01)

    plt.tight_layout()
    plt.show()
    
    return best_params

def voltage_to_flux_quanta(voltage, best_params):
    """
    Converts voltage to normalized flux quanta using the fitted parameters.

    Parameters:
    -----------
    voltage : float or array-like
        The voltage value(s).
    best_params : dict
        The dictionary containing the best-fit parameters from fit_resonance_data.

    Returns:
    --------
    flux_quanta : float or array-like
        The normalized flux quanta value(s).
    """
    if best_params is None:
        raise ValueError("Fit parameters are not available. Run fit_resonance_data first.")

    V_offset = best_params['V_offset']
    V_period = best_params['V_period']

    flux_quanta = (voltage - V_offset) / V_period
    return flux_quanta