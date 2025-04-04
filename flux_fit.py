import json
import os

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


def make_json_serializable(obj):
    """Convert numpy arrays, pandas DataFrames and other non-serializable objects to JSON serializable types."""
    import pandas as pd
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')  # Convert DataFrame to list of records
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def identify_flux_quanta(df, save_path=""):
    """
    Identify and fit individual flux quanta in the resonator frequency vs voltage dataset.
    Works for both single and multiple flux quanta.
    """
    import matplotlib.backends.backend_pdf
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    # Sort the dataframe by voltage to ensure proper peak finding
    df_sorted = df.sort_values('voltage').reset_index(drop=True)
    
    # Calculate prominence threshold from the data
    f_range = df_sorted['f'].max() - df_sorted['f'].min()
    prominence_threshold = f_range * 0.1
    
    # Find maxima in the frequency data
    maxima_indices, peak_properties = find_peaks(
        df_sorted['f'].values,
        distance=10,
        prominence=prominence_threshold
    )
    
    # Handle case where no maxima are found (likely single quantum)
    if len(maxima_indices) == 0:
        print("No clear maxima found. Treating as single flux quantum.")
        maxima_indices = np.array([np.argmax(df_sorted['f'].values)])
        avg_period = np.abs(df_sorted['voltage'].max() - df_sorted['voltage'].min())
    else:
        # Get the voltage values at the maxima
        maxima_voltages = df_sorted['voltage'].iloc[maxima_indices].values
        
        # Calculate the average period from voltage differences between peaks
        periods = np.diff(maxima_voltages)
        avg_period = np.mean(periods)
    
    maxima_voltages = df_sorted['voltage'].iloc[maxima_indices].values
    n_quanta = max(1, len(maxima_indices))
    
    print(f"Detected {n_quanta} flux quanta")
    print(f"Estimated period: {avg_period:.2f} mV")
    
    # Initialize PDF
    pdf_name = os.path.join(save_path, "flux_quanta_analysis.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    
    # Colors for different quanta
    n_quanta = len(maxima_indices)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_quanta))
    
    # Create the overview plot (Plot 1)
    fig1 = plt.figure(figsize=(12, 6))
    
    # Calculate boundaries
    all_boundaries = []
    if n_quanta == 1:
        # For single quantum, use the full voltage range
        all_boundaries = [
            df_sorted['voltage'].min(),
            df_sorted['voltage'].max()
        ]
    else:
        # For multiple quanta, calculate boundaries as before
        first_half_period = periods[0] / 2 if len(periods) > 0 else avg_period / 2
        all_boundaries.append(maxima_voltages[0] - first_half_period)
        
        for i in range(len(maxima_voltages) - 1):
            midpoint = (maxima_voltages[i] + maxima_voltages[i + 1]) / 2
            all_boundaries.append(midpoint)
        
        last_half_period = periods[-1] / 2 if len(periods) > 0 else avg_period / 2
        all_boundaries.append(maxima_voltages[-1] + last_half_period)
    
    # Store boundary information
    boundary_info = {
        'voltages': all_boundaries,
        'descriptions': ([f'Start of quantum 0'] + 
                       [f'Transition between quantum {i} and {i+1}' for i in range(n_quanta-1)] +
                       [f'End of quantum {n_quanta-1}']) if n_quanta > 1 else 
                      ['Start of quantum', 'End of quantum']
    }

    # Separate data into flux quanta and plot overview
    quanta = {}
    
    def fit_func(x, w0, q0, V_offset):
        return resonant_frequency(x, w0, q0, avg_period, V_offset)
    
    # Plot overview with colored quanta
    for i in range(len(all_boundaries) - 1):
        mask = (df_sorted['voltage'] >= all_boundaries[i]) & (df_sorted['voltage'] < all_boundaries[i + 1])
        voltage_data = df_sorted.loc[mask, 'voltage'].values
        freq_data = df_sorted.loc[mask, 'f'].values
        plt.scatter(voltage_data, freq_data, color=colors[i], alpha=0.5, label=f'Quantum {i}')
    
    # Add maxima points and boundaries to overview
    plt.scatter(maxima_voltages, df_sorted['f'].iloc[maxima_indices], 
                color='black', marker='x', s=100, label='Maxima')
    for boundary in all_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Overview: {n_quanta} Flux Quanta\nEstimated Period: {avg_period:.2f} mV')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the first plot
    pdf.savefig(fig1)
    
    # Create detailed subplots (Plot 2)
    fig2, axs = plt.subplots(n_quanta, 1, figsize=(12, 4*n_quanta))
    if n_quanta == 1:
        axs = [axs]
    
    # Process each quantum
    for i in range(len(all_boundaries) - 1):
        mask = (df_sorted['voltage'] >= all_boundaries[i]) & (df_sorted['voltage'] < all_boundaries[i + 1])
        voltage_data = df_sorted.loc[mask, 'voltage'].values
        freq_data = df_sorted.loc[mask, 'f'].values
        
        # Calculate normalized flux
        flux_data = (voltage_data - voltage_data[np.argmax(freq_data)]) / avg_period
        
        # Initial parameter guesses
        w0_guess = np.max(freq_data)
        q0_guess = 0.05
        V_offset_guess = voltage_data[np.argmax(freq_data)]
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(
                fit_func, 
                voltage_data, 
                freq_data,
                p0=[w0_guess, q0_guess, V_offset_guess],
                bounds=([w0_guess*0.95, 0.001, V_offset_guess-avg_period], 
                       [w0_guess*1.05, 0.2, V_offset_guess+avg_period])
            )
            
            # Generate smooth curve for the fit
            v_smooth = np.linspace(voltage_data.min(), voltage_data.max(), 100)
            f_smooth = fit_func(v_smooth, *popt)
            flux_smooth = (v_smooth - popt[2]) / avg_period
            
            # Store results
            quanta[f'quantum_{i}'] = {
                'voltage': voltage_data,
                'frequency': freq_data,
                'flux': flux_data,
                'voltage_range': (all_boundaries[i], all_boundaries[i + 1]),
                'best_fit': {
                    'w0': popt[0],
                    'q0': popt[1],
                    'V_period': avg_period,
                    'V_offset': popt[2],
                    'pcov': pcov
                },
                'fit_curve': {
                    'voltage': v_smooth,
                    'frequency': f_smooth,
                    'flux': flux_smooth
                }
            }
            
            # Plot in corresponding subplot
            ax = axs[i]
            
            # Plot data and fit
            ax.scatter(voltage_data, freq_data, color=colors[i], alpha=0.5, label='Data')
            ax.plot(v_smooth, f_smooth, '-', color=colors[i], alpha=0.8, label='Fit')
            
            # Create twin axis for flux
            ax2 = ax.twiny()
            ax2.set_xlim(min(flux_data), max(flux_data))
            
            # Labels and title
            ax.set_xlabel('Voltage (mV)')
            ax2.set_xlabel('Normalized Flux (Φ/Φ₀)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Quantum {i}\n' + 
                        f'w0 = {popt[0]:.3e} Hz, q0 = {popt[1]:.3f}\n' +
                        f'V_offset = {popt[2]:.2f} mV')
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"\nQuantum {i} fit parameters:")
            print(f"w0 = {popt[0]:.6e} Hz")
            print(f"q0 = {popt[1]:.6f}")
            print(f"V_offset = {popt[2]:.6f} mV")
            
        except RuntimeError as e:
            print(f"Failed to fit quantum {i}: {str(e)}")
            quanta[f'quantum_{i}'] = {
                'voltage': voltage_data,
                'frequency': freq_data,
                'flux': flux_data,
                'voltage_range': (all_boundaries[i], all_boundaries[i + 1]),
                'best_fit': None
            }
    
    plt.tight_layout()
    
    # Save the second plot
    pdf.savefig(fig2)
    
    # Close the PDF file
    pdf.close()
    
    # Show the plots
    plt.show()
    
    results = {
        'n_periods': n_quanta,
        'maxima_indices': maxima_indices,
        'maxima_voltages': maxima_voltages,
        'period_estimate': avg_period,
        'df_sorted': df_sorted,
        'quanta': quanta,
        'boundaries': boundary_info,
        'is_single_quantum': n_quanta == 1
    }
    
    # For saving to a file in the specified directory:
    if os.path.isdir(save_path):
        # If save_path is a directory, save the JSON in that directory
        json_name = os.path.join(save_path, "flux_quanta_analysis.json")
    else:
        # If save_path is a file path (e.g. for PDF), use its directory for JSON
        json_dir = os.path.dirname(save_path)
        if not json_dir:  # If there's no directory part
            json_dir = '.'
        json_name = os.path.join(json_dir, "flux_quanta_analysis.json")

    # Remove the DataFrame from results before serializing
    results_for_json = {k: v for k, v in results.items() if k != 'df_sorted'}
    serializable_results = make_json_serializable(results_for_json)
    with open(json_name, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Return full results including DataFrame
    return results


def reject_frequency_outliers(df, freq_threshold=5.79e9):
    """
    Simple anomaly rejection that removes frequency points above a threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing resonator measurements with 'f' column
    freq_threshold : float, optional
        Frequency threshold above which points are considered outliers (default: 5.79e9)
    
    Returns:
    --------
    tuple
        (cleaned_df, mask, stats)
    """
    # Create mask for points below threshold
    mask = df['f'] < freq_threshold
    
    # Create stats dictionary
    stats = {
        'total_initial_points': len(df),
        'rejected_points': (~mask).sum(),
        'remaining_points': mask.sum()
    }
    
    # Create cleaned dataframe
    cleaned_df = df[mask].copy()
    
    return cleaned_df, mask, stats

def load_flux_quanta_analysis(json_path):
    """
    Load flux quanta analysis results from a JSON file.
    
    Parameters:
    -----------
    json_path : str
        Path to the flux quanta analysis JSON file
        
    Returns:
    --------
    dict
        Dictionary containing the analysis results
    """
    import json

    import numpy as np

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays for numerical data
    if 'maxima_voltages' in data:
        data['maxima_voltages'] = np.array(data['maxima_voltages'])
    
    if 'maxima_indices' in data:
        data['maxima_indices'] = np.array(data['maxima_indices'])
    
    # Convert the quanta data back to numpy arrays
    if 'quanta' in data:
        for quantum_key, quantum_data in data['quanta'].items():
            if 'voltage' in quantum_data:
                quantum_data['voltage'] = np.array(quantum_data['voltage'])
            if 'frequency' in quantum_data:
                quantum_data['frequency'] = np.array(quantum_data['frequency'])
            if 'flux' in quantum_data:
                quantum_data['flux'] = np.array(quantum_data['flux'])
            
            # Convert fit curve data to numpy arrays
            if 'fit_curve' in quantum_data:
                for key, value in quantum_data['fit_curve'].items():
                    quantum_data['fit_curve'][key] = np.array(value)
            
            # Convert best fit parameters
            if 'best_fit' in quantum_data and quantum_data['best_fit'] is not None:
                if 'pcov' in quantum_data['best_fit']:
                    quantum_data['best_fit']['pcov'] = np.array(quantum_data['best_fit']['pcov'])
    
    # Convert boundary values to numpy arrays
    if 'boundaries' in data and 'voltages' in data['boundaries']:
        data['boundaries']['voltages'] = np.array(data['boundaries']['voltages'])
    
    print(f"Loaded flux quanta analysis with {data['n_periods']} periods")
    print(f"Period estimate: {data['period_estimate']:.2f} mV")
    
    return data

def phi_to_voltage_frequency(flux_data, phi_values):
    """
    Convert normalized flux (phi) values to voltage and frequency using the 
    fitted parameters for each flux quantum.
    
    Parameters:
    -----------
    flux_data : dict
        Flux quanta analysis results loaded from JSON
    phi_values : float or list/array of floats
        Phi value(s) to convert
        
    Returns:
    --------
    dict
        Dictionary with results for each quantum
    """
    import numpy as np

    # Ensure phi_values is an array
    if np.isscalar(phi_values):
        phi_values = [phi_values]
    else:
        phi_values = np.array(phi_values)
    
    results = {}
    
    for quantum_key, quantum_data in flux_data['quanta'].items():
        # Skip if no best fit
        if 'best_fit' not in quantum_data or quantum_data['best_fit'] is None:
            continue
            
        # Get fit parameters
        best_fit = quantum_data['best_fit']
        w0 = best_fit['w0']
        q0 = best_fit['q0']
        V_period = best_fit['V_period']
        V_offset = best_fit['V_offset']
        
        # Calculate voltages for the phi values
        voltages = phi_values * V_period + V_offset
        
        # Calculate frequencies using the resonant_frequency function
        frequencies = []
        for phi in phi_values:
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
            safe_denominator = denominator if abs(denominator) > 1e-10 else 1e-10 * np.sign(denominator)
            
            # Calculate the frequency
            exponent = -0.5
            frequency = w0 * (1 + q0 * numerator / safe_denominator) ** exponent
            frequencies.append(frequency)
        
        # Store results for this quantum
        results[quantum_key] = {
            "phi": phi_values,
            "voltage": voltages,
            "frequency": frequencies
        }
    
    return results

def process_phi_results(phi_results):
    guess_data = {}
    for quantum_key, results in phi_results.items():
        print(f"\n{quantum_key}:")
        for i, phi in enumerate(results['phi']):
            print(f"  φ = {phi:.2f} → Voltage = {results['voltage'][i]:.2f} mV → Frequency = {results['frequency'][i]:.3e} Hz")
            guess_data[quantum_key] = {'voltage' : results['voltage'][i], 'frequency' : results['frequency'][i], 'phi' : phi}
    return guess_data

def plot_phi_points(flux_data, phi_values):
    """
    Plot the original data, fits, and user-specified phi points for each quantum.
    
    Parameters:
    -----------
    flux_data : dict
        Flux quanta analysis results loaded from JSON
    phi_values : float or list/array of floats
        Phi value(s) to highlight on the plots
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate voltage and frequency for each phi value
    phi_results = phi_to_voltage_frequency(flux_data, phi_values)
    
    # Create overview plot
    plt.figure(figsize=(12, 6))
    
    # Plot boundaries
    boundaries = flux_data['boundaries']['voltages']
    for boundary in boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Colors for different quanta
    colors = plt.cm.rainbow(np.linspace(0, 1, flux_data['n_periods']))
    
    # Plot each quantum
    for i, (quantum_key, quantum_data) in enumerate(flux_data['quanta'].items()):
        # Plot data points
        plt.scatter(quantum_data['voltage'], quantum_data['frequency'], 
                   color=colors[i], alpha=0.5, label=f'Quantum {quantum_key[-1]}')
        
        # Plot fit curve if available
        if 'fit_curve' in quantum_data:
            plt.plot(quantum_data['fit_curve']['voltage'], 
                    quantum_data['fit_curve']['frequency'], 
                    '-', color=colors[i], alpha=0.8)
    
    # Plot the specified phi points for each quantum
    for i, (quantum_key, results) in enumerate(phi_results.items()):
        plt.scatter(results['voltage'], results['frequency'], 
                   marker='*', s=150, color=colors[i], edgecolor='black',
                   label=f'Phi points {quantum_key[-1]}')
        
        # Add text labels for phi values
        for j, (phi, v, f) in enumerate(zip(results['phi'], results['voltage'], results['frequency'])):
            plt.annotate(f'φ={phi:.2f}', 
                        (v, f), 
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Flux Quanta with User-Specified Phi Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return phi_results
