import logging
import os
import shutil
import subprocess
import time

from dotenv import load_dotenv

drive_path = "Z:\\"
PATH_TO_EXE = "C:\\Users\\james\\Documents\\Quasiparticle\\Quasiparticle.exe"

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
