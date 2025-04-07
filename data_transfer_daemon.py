import fcntl
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Load environment variables
load_dotenv()

class DataTransferHandler(FileSystemEventHandler):
    def __init__(self):
        self.source_path = os.getenv('LOCAL_STORAGE_PATH')
        self.hpc_user = os.getenv('HPC_USERNAME')
        self.hpc_host = os.getenv('HPC_HOSTNAME')
        self.destination_path = os.getenv('HPC_BASE_PATH')  # This is now the full path
        self.project_name = os.getenv('PROJECT_NAME')
        self.device_name = os.getenv('DEVICE_NAME')
        self.batch_size = 1  # Process files immediately instead of batching
        self.last_process_time = time.time()
        self.process_interval = 0.5  # Process at least every 500ms
        
        if not all([self.source_path, self.hpc_user, self.hpc_host, 
                   self.destination_path]):
            raise ValueError("Missing required environment variables")
        
        # Now we use destination_path directly since it's the full path
        self.hpc_experiment_path = self.destination_path
        
        logging.info(f"Local source path: {self.source_path}")
        logging.info(f"HPC path: {self.hpc_experiment_path}")
        
        self.pending_files = {}
        
        # Setup logging with absolute path
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_transfer.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        
        # Verify HPC directory exists
        self._ensure_remote_directory()
        
        # State tracking
        self.transfer_history_file = os.path.join(self.source_path, '.transfer_history.json')
        self.lock_file = os.path.join(self.source_path, '.transfer.lock')
        
        # Load transfer history
        self.transfer_history = self._load_transfer_history()
        
        # Ensure single instance
        self._acquire_lock()
        
        # Initial connection test
        self._test_hpc_connection()

    def _acquire_lock(self):
        """Ensure only one instance is running"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            raise RuntimeError("Another instance of the transfer daemon is already running")
        
    def _load_transfer_history(self):
        """Load transfer history from JSON file"""
        if os.path.exists(self.transfer_history_file):
            try:
                with open(self.transfer_history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning("Corrupt transfer history file, creating new one")
        return {}

    def _save_transfer_history(self):
        """Save transfer history to JSON file"""
        with open(self.transfer_history_file, 'w') as f:
            json.dump(self.transfer_history, f, indent=2)

    def _test_hpc_connection(self):
        """Test HPC connection and SSH setup with detailed error reporting"""
        try:
            # Test basic connection with verbose output
            test_cmd = f"ssh -v {self.hpc_user}@{self.hpc_host} echo 'test'"
            result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logging.error("SSH Connection Test Details:")
                logging.error(f"Command: {test_cmd}")
                logging.error(f"Return Code: {result.returncode}")
                logging.error(f"STDOUT: {result.stdout}")
                logging.error(f"STDERR: {result.stderr}")
                
                # Check common SSH issues
                if "Permission denied" in result.stderr:
                    logging.error("SSH key authentication failed. Please check your SSH keys")
                elif "Could not resolve hostname" in result.stderr:
                    logging.error("Could not resolve HPC hostname. Check your .env file and network connection")
                elif "Connection refused" in result.stderr:
                    logging.error("Connection was refused. Check if SSH service is running on HPC")
                elif "Connection timed out" in result.stderr:
                    logging.error("Connection timed out. Check your network connection and HPC status")
                    
                raise RuntimeError("SSH connection test failed")
                
            logging.info("SSH connection test successful")
            
        except subprocess.TimeoutExpired:
            logging.error("SSH connection test timed out after 30 seconds")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during SSH test: {str(e)}")
            raise

    def _is_file_ready(self, filepath, wait_time=5):
        """
        Ensure file is completely written and not being modified
        Returns True if file is ready for transfer
        """
        try:
            initial_size = os.path.getsize(filepath)
            initial_mtime = os.path.getmtime(filepath)
            time.sleep(wait_time)  # Wait to ensure file is not being written
            return (initial_size == os.path.getsize(filepath) and 
                   initial_mtime == os.path.getmtime(filepath))
        except OSError:
            return False

    def _verify_transfer(self, local_path, remote_path, max_retries=3):
        """Enhanced verification with retries"""
        for attempt in range(max_retries):
            try:
                local_checksum = self._get_file_checksum(local_path)
                remote_checksum = self._get_remote_checksum(remote_path)
                
                if not all([local_checksum, remote_checksum]):
                    logging.error(f"Checksum calculation failed on attempt {attempt + 1}")
                    continue
                
                if local_checksum == remote_checksum:
                    # Add to transfer history
                    self.transfer_history[local_path] = {
                        'timestamp': datetime.now().isoformat(),
                        'checksum': local_checksum,
                        'remote_path': remote_path,
                        'verified': True
                    }
                    self._save_transfer_history()
                    return True
                
                logging.error(f"Checksum mismatch on attempt {attempt + 1}")
            except Exception as e:
                logging.error(f"Verification error on attempt {attempt + 1}: {str(e)}")
            
            time.sleep(5)  # Wait before retry
        return False

    def process_pending_files(self):
        """Process pending file transfers with detailed logging"""
        if not self.pending_files:
            return
            
        logging.info(f"Processing {len(self.pending_files)} pending files")
        
        # Test connection before starting transfers
        try:
            logging.info("Testing HPC connection before transfer...")
            self._test_hpc_connection()
        except Exception as e:
            logging.error(f"HPC connection failed, skipping transfers: {str(e)}")
            return
            
        files_to_transfer = self.pending_files.copy()
        
        for bin_path, file_info in files_to_transfer.items():
            txt_path = file_info['txt_path']
            rel_dir = file_info['relative_dir']
            
            logging.info(f"\nProcessing file pair:")
            logging.info(f"Binary file: {bin_path}")
            logging.info(f"Text file: {txt_path}")
            logging.info(f"Relative directory: {rel_dir}")
            
            remote_dir = os.path.join(self.hpc_experiment_path, rel_dir)
            logging.info(f"Remote directory will be: {remote_dir}")
            
            try:
                # Convert bin to npz
                npz_path = self._convert_bin_to_npz(bin_path)
                
                # Create remote directory
                mkdir_cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'mkdir -p {remote_dir}'"
                logging.info(f"Creating remote directory with command: {mkdir_cmd}")
                
                result = subprocess.run(mkdir_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"Failed to create remote directory:")
                    logging.error(f"Return code: {result.returncode}")
                    logging.error(f"STDOUT: {result.stdout}")
                    logging.error(f"STDERR: {result.stderr}")
                    continue
                
                transfer_success = True
                # Transfer NPZ and TXT files
                for local_file in [npz_path, txt_path]:
                    filename = os.path.basename(local_file)
                    remote_path = f"{self.hpc_user}@{self.hpc_host}:{os.path.join(remote_dir, filename)}"
                    remote_full_path = os.path.join(remote_dir, filename)
                    
                    logging.info(f"\nTransferring {local_file} to {remote_path}")
                    
                    # Use rsync with verbose output
                    rsync_cmd = f"rsync -avz --progress {local_file} {remote_path}"
                    logging.info(f"Transfer command: {rsync_cmd}")
                    
                    result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        logging.error("Transfer failed:")
                        logging.error(f"Return code: {result.returncode}")
                        logging.error(f"STDOUT: {result.stdout}")
                        logging.error(f"STDERR: {result.stderr}")
                        transfer_success = False
                        break
                    
                    logging.info("Transfer completed, verifying...")
                    if not self._verify_transfer(local_file, remote_full_path):
                        logging.error(f"Verification failed for {local_file}")
                        transfer_success = False
                        break
                    
                    logging.info("Verification successful")
                
                if transfer_success:
                    logging.info("All files transferred successfully, cleaning up...")
                    # Remove original binary file, NPZ file, and txt file
                    for local_file in [bin_path, npz_path, txt_path]:
                        if os.path.exists(local_file):
                            os.remove(local_file)
                            logging.info(f"Removed verified file: {local_file}")
                    
                    del self.pending_files[bin_path]
                    self._cleanup_empty_directories(bin_path)
                    
            except Exception as e:
                logging.error(f"Error processing {bin_path}: {str(e)}")
                continue

    def _cleanup_empty_directories(self, start_path):
        """Safely clean up empty directories"""
        dir_path = os.path.dirname(start_path)
        while dir_path != self.source_path:
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    logging.info(f"Removed empty directory: {dir_path}")
                dir_path = os.path.dirname(dir_path)
            except Exception as e:
                logging.error(f"Error cleaning up directory {dir_path}: {str(e)}")
                break

    def _ensure_remote_directory(self):
        """Verify the HPC directory exists and is writable"""
        try:
            # Check if directory exists
            check_cmd = f"ssh {self.hpc_user}@{self.hpc_host} '[ -d {self.hpc_experiment_path} ]'"
            result = subprocess.run(check_cmd, shell=True)
            
            if result.returncode != 0:
                logging.error(f"HPC directory {self.hpc_experiment_path} does not exist or is not accessible")
                raise RuntimeError(f"HPC directory {self.hpc_experiment_path} is not accessible")
            
            # Verify write permissions with a test file
            test_cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'touch {self.hpc_experiment_path}/.test && rm {self.hpc_experiment_path}/.test'"
            subprocess.run(test_cmd, shell=True, check=True)
            logging.info(f"Verified write access to: {self.hpc_experiment_path}")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to verify HPC directory: {e}")
            raise RuntimeError(f"Failed to verify HPC directory: {e}")
    
    def on_created(self, event):
        """Handle file creation events immediately"""
        if event.is_directory:
            return
            
        filepath = event.src_path
        current_time = time.time()
        
        # Only process .bin files and their companion .txt files
        if filepath.endswith('.bin'):
            txt_path = filepath[:-4] + '.txt'
            if os.path.exists(txt_path):
                self._add_to_pending(filepath, txt_path)
                # Process immediately if enough time has passed
                if current_time - self.last_process_time >= self.process_interval:
                    self.process_pending_files()
                    self.last_process_time = current_time
        elif filepath.endswith('.txt'):
            bin_path = filepath[:-4] + '.bin'
            if os.path.exists(bin_path):
                self._add_to_pending(bin_path, filepath)
                # Process immediately if enough time has passed
                if current_time - self.last_process_time >= self.process_interval:
                    self.process_pending_files()
                    self.last_process_time = current_time
    
    def _add_to_pending(self, bin_path, txt_path):
        """Add a file pair to pending transfers."""
        rel_path = os.path.relpath(bin_path, self.source_path)
        dir_path = os.path.dirname(rel_path)
        
        self.pending_files[bin_path] = {
            'txt_path': txt_path,
            'relative_dir': dir_path
        }
        
        logging.info(f"Added to pending transfers: {rel_path}")
    
    def _get_file_checksum(self, filepath):
        """Calculate SHA-256 checksum of a local file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_remote_checksum(self, remote_path):
        """Get SHA-256 checksum of a remote file"""
        cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'sha256sum {remote_path}'"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            # sha256sum output format: "<hash>  <filepath>"
            return result.stdout.split()[0]
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to get remote checksum: {e}")
            return None
    def _convert_bin_to_npz(self, bin_fpath):
        """
        Converts a .bin file (Alazar data) to a compressed .npz file.

        Args:
            bin_fpath (str): Path to the input .bin file.
        """
        if bin_fpath[-4:] != '.bin':
            print(f'Warning: Input fpath "{bin_fpath}" should be a .bin file.')
            return False

        try:
            data = np.fromfile(bin_fpath, dtype=np.uint16)
            npz_fpath = bin_path[:-4] + '.npz'
            np.savez_compressed(npz_fpath, data=data)  # Save the array with the key 'data'
            print(f"Successfully converted '{bin_fpath}' to '{npz_fpath}'")
            return npz_fpath
        except Exception as e:
            print(f"Error converting '{bin_fpath}' to '{npz_fpath}': {e}")
            return None

    def _convert_bin_to_npz_v1(self, bin_path):
        """
        Convert binary data file to NPZ format.
        
        Parameters:
        -----------
        bin_path : str
            Path to the binary file
            
        Returns:
        --------
        str
            Path to the created NPZ file
        """
        try:
            # Read binary data
            DATA = np.fromfile(bin_path, dtype=np.uint16)
            # Convert to voltage
            DATA = (DATA - 2047.5) * 0.4/2047.5
            
            # Get metadata from companion txt file
            txt_path = bin_path[:-4] + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    metadata_lines = f.read().splitlines()
                    # Parse metadata for channels info
                    channels_info = next((line for line in metadata_lines if "Channels:" in line), None)
                    if channels_info and "AB" in channels_info:
                        # Split data into two channels
                        DATA = [DATA[ind::2] for ind in range(2)]
            
            # Create NPZ file path
            npz_path = bin_path[:-4] + '.npz'
            
            # Save as NPZ
            if isinstance(DATA, list):
                # For two-channel data
                np.savez_compressed(npz_path, 
                                  channel_A=DATA[0],
                                  channel_B=DATA[1])
            else:
                # For single-channel data
                np.savez_compressed(npz_path, 
                                  data=DATA)
            
            logging.info(f"Successfully converted {bin_path} to {npz_path}")
            return npz_path
        
        except Exception as e:
            logging.error(f"Error converting {bin_path} to NPZ: {str(e)}")
            raise

def run_watcher():
    """Run the file watcher with more frequent checks"""
    handler = DataTransferHandler()
    observer = Observer()
    observer.schedule(handler, handler.source_path, recursive=True)
    observer.start()
    
    logging.info(f"Started watching: {handler.source_path}")
    logging.info(f"Transferring to: {handler.hpc_experiment_path}")
    
    try:
        while True:
            # Process files more frequently
            handler.process_pending_files()
            time.sleep(0.1)  # Check every 100ms instead of 1 second
    except KeyboardInterrupt:
        logging.info("Stopping file watcher")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    run_watcher()