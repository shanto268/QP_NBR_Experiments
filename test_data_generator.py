import logging
import os
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

from transfer_utils import (check_storage_space, start_transfer_daemon,
                            stop_transfer_daemon, verify_hpc_connection)


def create_dummy_data(size_mb=100):
    """Create dummy binary data of specified size in MB"""
    return np.random.bytes(size_mb * 1024 * 1024)

def write_dummy_metadata(file_path, phi, f_drive, voltage, T_MXC, T_Rad, f_clearing, P_clearing):
    """Write metadata matching your experiment format"""
    with open(file_path, 'w') as f:
        f.write("=== Basic Info ===\n")
        f.write(f"Timestamp: {time.strftime('%c')}\n")
        f.write(f"Digital Attenuator: {np.random.randint(14, 30)} dB\n")
        f.write(f"Sample rate: 300 MHz\n")
        f.write(f"Acquisition length: 5 seconds\n")
        f.write(f"Drive frequency: {f_drive:.6f} GHz\n")
        f.write(f"Temperature MXC: {T_MXC} mK\n")
        f.write(f"Radiator temperature: {T_Rad} mK\n")
        f.write(f"Flux bias (Phi): {phi:.6f} Phi0\n")
        f.write(f"Flux voltage: {voltage*1e3:.6f} mV\n")
        f.write(f"Clearing frequency: {f_clearing:.6f} GHz\n")
        f.write(f"Clearing power: {P_clearing:.6f} dBm\n")
        f.write("=== End Basic Info ===\n")

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def verify_transfer_status(base_path):
    """Verify all files have been transferred"""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.bin') or file.endswith('.txt'):
                full_path = os.path.join(root, file)
                log_with_timestamp(f"File still exists locally: {full_path}")
                return False
    return True

def run_experiment():
    """Run the experiment with integrated data transfer"""
    load_dotenv()
    base_path = os.getenv('LOCAL_STORAGE_PATH')
    if not base_path:
        raise ValueError("LOCAL_STORAGE_PATH not set in .env")

    # Verify HPC connection before starting
    log_with_timestamp("Verifying HPC connection...")
    verify_hpc_connection()

    # Start the daemon
    log_with_timestamp("Starting data transfer daemon...")
    daemon_process = start_transfer_daemon()
    
    try:
        # Experiment parameters
        phi_arr = np.linspace(-0.5, -0.3, 3)
        f_clearing_arr = np.arange(5, 8, 3)
        P_clearing_arr = np.arange(-10, -7, 3)
        
        # Fixed parameters
        T_MXC = 26  # mK
        T_Rad = 1.8  # mK
        detuning = 0.5e6  # Hz
        
        total_points = len(phi_arr) * len(f_clearing_arr) * len(P_clearing_arr)
        log_with_timestamp(f"Starting experiment with {total_points} total points")
        log_with_timestamp(f"Data will be saved to: {base_path}")
        
        start_time = time.time()
        
        for phi_idx, phi in enumerate(phi_arr):
            log_with_timestamp(f"\nProcessing phi = {phi:.3f} ({phi_idx+1}/{len(phi_arr)})")
            
            # Check storage space periodically
            check_storage_space(base_path, required_gb=20)
            
            # Simulate finding resonance frequency
            f_phi = 5.7 + 0.1 * np.sin(phi)
            f_drive = f_phi - detuning * 1e-9
            voltage = phi * 20
            
            for f_clearing in f_clearing_arr:
                for P_clearing in P_clearing_arr:
                    phi_str = f"phi_{phi:.3f}".replace('.', 'p').strip()
                    da_value = np.random.randint(14, 30)
                    clearing_str = f"clearing_{f_clearing:.2f}GHz_{P_clearing:.1f}dBm".replace('.', 'p')
                    
                    dir_path = os.path.join(
                        base_path,
                        phi_str,
                        f"DA{da_value:02d}_SR10",
                        clearing_str
                    )
                    
                    os.makedirs(dir_path, exist_ok=True)
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    time.sleep(1)
                    
                    bin_file = os.path.join(dir_path, f"test_device_{timestamp}.bin")
                    txt_file = bin_file[:-4] + '.txt'
                    
                    with open(bin_file, 'wb') as f:
                        f.write(create_dummy_data(size_mb=1))
                    
                    write_dummy_metadata(
                        txt_file, phi, f_drive, voltage, 
                        T_MXC, T_Rad, f_clearing, P_clearing
                    )
                    
                    log_with_timestamp(
                        f"Created files for phi={phi:.3f}, "
                        f"f_clearing={f_clearing}GHz, P_clearing={P_clearing}dBm"
                    )
                    
                    # Check if previous files were transferred
                    previous_dir = os.path.dirname(bin_file)
                    if os.path.exists(previous_dir):
                        files = os.listdir(previous_dir)
                        if not files:
                            log_with_timestamp(f"Previous directory cleaned up: {previous_dir}")
                    
                    time.sleep(0.5)  # Give daemon time to process
            
            # Progress update
            elapsed_time = time.time() - start_time
            points_done = (phi_idx + 1) * len(f_clearing_arr) * len(P_clearing_arr)
            points_remaining = total_points - points_done
            time_per_point = elapsed_time / points_done
            remaining_time = points_remaining * time_per_point
            
            log_with_timestamp(f"Completed {points_done}/{total_points} points")
            log_with_timestamp(f"Estimated remaining time: {remaining_time/60:.1f} minutes")
            
        # After experiment completion, wait for transfers
        log_with_timestamp("\nWaiting for all transfers to complete...")
        max_wait = 300  # 5 minutes
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            if verify_transfer_status(base_path):
                log_with_timestamp("All files transferred successfully!")
                break
            log_with_timestamp("Waiting for transfers to complete...")
            time.sleep(10)
        else:
            log_with_timestamp("WARNING: Some files may not have been transferred!")
            
    except KeyboardInterrupt:
        log_with_timestamp("\nExperiment interrupted by user")
    except Exception as e:
        log_with_timestamp(f"\nError during experiment: {str(e)}")
        raise
    finally:
        # Give daemon time to process final files
        log_with_timestamp("Waiting for final transfers to complete...")
        time.sleep(5)  # Wait a bit for last files
        
        # Stop the daemon gracefully
        log_with_timestamp("Stopping data transfer daemon...")
        stop_transfer_daemon(daemon_process)
        
        total_time = time.time() - start_time
        log_with_timestamp(f"\nExperiment finished in {total_time/60:.1f} minutes")

if __name__ == "__main__":
    log_with_timestamp("Starting experiment...")
    run_experiment()
