# QP Experiments

## Table of Contents

- [Experiment 0: Flux Sweeping](#experiment-0-flux-sweeping)
- [Experiment 1: IQ Data Acquisition with Flux and Clearing Tone Variation](#experiment-1-acquiring-time-domain-iq-data-following-the-flux-curve-and-varying-the-clearing-tone-frequency-and-power)
  - [Set Up](#set-up)
    - [HPC](#hpc)
    - [Measurements](#measurements)
- [Experiment 2: IQ Data Acquisition](#experiment-2-acquiring-time-domain-iq-data)
- [To Do](#to-do)

<details>
<summary><h2>Experiment 0: Flux Sweeping</h2></summary>

A notebook that uses the SRS voltage source to sweep flux and record the resonator responce using the VNA.

**notebook name:** [experiment0.ipynb](experiment0.ipynb)

</details>

<details>
<summary><h2>Experiment 1: Acquiring time-domain IQ data following the flux curve and varying the clearing tone frequency and power</h2></summary>

**notebook name:** [experiment1.ipynb](experiment1.ipynb)

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

</details>

<details>
<summary><h2>Set Up</h2></summary>

### HPC:

Make sure you have SSH access to your HPC cluster
Set up SSH key-based authentication to avoid password prompts
Fill in the .env file with your HPC details

### Measurements:

1. Flux tuning curve of the resonator

---

## To Do:

- [ ] Test the rsync HPC back up mechanism
- [ ] Add code to dynamically change the frequency search range based on multiple flux quanta fit data
- [ ] Add feature to change .bin to a lower storage fingerprint file as a dameon
- [ ] Add resonator circle fit code
- [ ] Add the dynamic driving next mode code
- [ ] Add feature to upload saved data to HPC via Globus API

</details>

<details>
<summary><h2>Experiment 2: Acquiring time-domain IQ data</h2></summary>

A notebook that sets the LO at user specified parameters and takes the time domain IQ data with the Alazar card.

**notebook name:** [experiment2.ipynb](experiment2.ipynb)

</details>

---
