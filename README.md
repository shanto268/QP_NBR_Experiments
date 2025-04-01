# QP Experiments

## Experiment 1: Acquiring time-domain IQ data following the flux curve and varying the clearing tone frequency and power

High level overview:

```python
v_arr, phi_arr, f_arr = from_flux_fit()
detuning = 0 # GHz
f_clearing_arr, P_clearing_arr = [], []

def find_mapped_resonance(phi):
    global phi_arr, f_arr
    phi_index = phi_arr[phi]
    return f_arr[phi_index]

def find_resonance(phi, span):
    f_guess = find_mapped_resonance(phi)
    set_vna(f_guess, span)
    f_phi = fit_vna_trace()
    turn_off_vna()
    return f_phi

for phi in phi_arr:

    voltage = get_voltage(phi)
    set_srs(voltage)
    f_phi = find_resonance(phi)
    f_drive = f_phi - detuning
    set_drive_tone(f_drive)

    for f_clearing in f_clearing_arr:
        for P_clearing in P_clearing_arr:
            set_clearing_tone(f_clearing, P_clearing)
            acquire_IQ_data()
            wait()


```

---

## To Do:

- [ ] Add feature to upload saved data to HPC via Globus and clear out local files safely
- [ ] Add feature to change .bin to a lower storage fingerprint file
- [ ] Add resonator circle fit code
- [ ] Add the dynamic driving next mode code
- [ ] Add code to dynamically change the frequency search range based on multiple flux quanta fit data
