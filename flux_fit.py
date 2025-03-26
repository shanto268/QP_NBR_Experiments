import matplotlib.pyplot as plt
import numpy as np


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

def flux_quanta_to_frequency(flux_quanta, best_params):
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
    return frequency

def phi_to_voltage(phi, best_params):
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
    return voltage

def from_flux_fit(phi_arr, best_fit):
    v_arr = phi_to_voltage(phi_arr, best_fit)
    f_arr = flux_quanta_to_frequency(phi_arr, best_fit)
    return v_arr, f_arr

def find_mapped_resonance(phi, best_fit):
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
    f_arr = flux_quanta_to_frequency(phi, best_fit)
    return f_arr

def plot_sweep_points(v_arr, f_arr, phi_arr):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(v_arr, f_arr, ".")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Frequency (GHz)")
    plt.title("Flux Points to be swept")

    plt.subplot(1, 2, 2)
    plt.plot(phi_arr, f_arr, ".")
    plt.xlabel("Phi")
    plt.ylabel("Frequency (GHz)")
    plt.title("Phi Points to be swept")

    plt.tight_layout()
    plt.show()