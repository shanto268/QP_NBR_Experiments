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

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Load environment variables
load_dotenv()

class DataTransferHandler(FileSystemEventHandler):
    def __init__(self):
        # Basic setup
        self.source_path = os.getenv('LOCAL_STORAGE_PATH')
        self.hpc_user = os.getenv('HPC_USERNAME')
        self.hpc_host = os.getenv('HPC_HOSTNAME')
        self.destination_path = os.getenv('HPC_BASE_PATH')
        self.project_name = os.getenv('PROJECT_NAME')
        self.device_name = os.getenv('DEVICE_NAME')
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # Validation
        if not all([self.source_path, self.hpc_user, self.hpc_host, 
                   self.destination_path, self.project_name, self.device_name]):
            raise ValueError("Missing required environment variables")
        
        # Create paths
        self.hpc_experiment_path = os.path.join(
            self.destination_path,
            self.project_name,
            self.device_name
        )
        
        # State tracking
        self.pending_files = {}
        self.transfer_history_file = os.path.join(self.source_path, '.transfer_history.json')
        self.lock_file = os.path.join(self.source_path, '.transfer.lock')
        
        # Setup logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            filename=os.path.join(self.source_path, 'data_transfer.log')
        )
        
        # Load transfer history
        self.transfer_history = self._load_transfer_history()
        
        # Ensure single instance
        self._acquire_lock()
        
        # Initial connection test
        self._test_hpc_connection()
        
        # Create remote directory structure
        self._ensure_remote_directory()

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
        """Test HPC connection and SSH setup"""
        try:
            # Test basic connection
            test_cmd = f"ssh -q {self.hpc_user}@{self.hpc_host} echo 'test'"
            subprocess.run(test_cmd, shell=True, check=True, timeout=10)
            
            # Test write permissions
            test_dir = os.path.join(self.hpc_experiment_path, '.test')
            test_file = os.path.join(test_dir, 'test.txt')
            cmds = [
                f"mkdir -p {test_dir}",
                f"touch {test_file}",
                f"rm {test_file}",
                f"rmdir {test_dir}"
            ]
            for cmd in cmds:
                subprocess.run(f"ssh {self.hpc_user}@{self.hpc_host} '{cmd}'", 
                             shell=True, check=True, timeout=10)
        except Exception as e:
            raise RuntimeError(f"HPC connection test failed: {str(e)}")

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
        """Enhanced file processing with better error handling"""
        if not self.pending_files:
            return
            
        # Test connection before starting transfers
        try:
            self._test_hpc_connection()
        except Exception as e:
            logging.error(f"HPC connection failed, skipping transfers: {str(e)}")
            return
            
        files_to_transfer = self.pending_files.copy()
        
        for bin_path, file_info in files_to_transfer.items():
            txt_path = file_info['txt_path']
            rel_dir = file_info['relative_dir']
            
            # Skip if files are still being written
            if not all(self._is_file_ready(f) for f in [bin_path, txt_path]):
                continue
            
            remote_dir = os.path.join(self.hpc_experiment_path, rel_dir)
            
            try:
                # Ensure remote directory exists
                mkdir_cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'mkdir -p {remote_dir}'"
                subprocess.run(mkdir_cmd, shell=True, check=True, timeout=30)
                
                transfer_success = True
                for local_file in [bin_path, txt_path]:
                    filename = os.path.basename(local_file)
                    remote_path = f"{self.hpc_user}@{self.hpc_host}:{os.path.join(remote_dir, filename)}"
                    remote_full_path = os.path.join(remote_dir, filename)
                    
                    # Check if file already exists on remote
                    check_cmd = f"ssh {self.hpc_user}@{self.hpc_host} '[ -f {remote_full_path} ]'"
                    if subprocess.run(check_cmd, shell=True).returncode == 0:
                        logging.warning(f"File already exists on remote: {remote_full_path}")
                        continue
                    
                    # Transfer with rsync
                    rsync_cmd = f"rsync -av --progress {local_file} {remote_path}"
                    subprocess.run(rsync_cmd, shell=True, check=True, timeout=3600)  # 1 hour timeout
                    
                    # Verify transfer
                    if not self._verify_transfer(local_file, remote_full_path):
                        transfer_success = False
                        break
                
                if transfer_success:
                    # Only remove local files after successful transfer and verification
                    for local_file in [bin_path, txt_path]:
                        if os.path.exists(local_file):
                            # Keep a backup of the metadata
                            if local_file.endswith('.txt'):
                                with open(local_file, 'r') as f:
                                    self.transfer_history[local_file + '_content'] = f.read()
                            os.remove(local_file)
                            logging.info(f"Removed verified file: {local_file}")
                    
                    del self.pending_files[bin_path]
                    self._save_transfer_history()
                    
                    # Clean up empty directories
                    self._cleanup_empty_directories(bin_path)
                    
            except Exception as e:
                logging.error(f"Transfer failed for {bin_path}: {str(e)}")
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
        """Ensure the remote directory structure exists"""
        cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'mkdir -p {self.hpc_experiment_path}'"
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"Ensured remote directory exists: {self.hpc_experiment_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create remote directory: {e}")
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        filepath = event.src_path
        
        # Only process .bin files and their companion .txt files
        if filepath.endswith('.bin'):
            txt_path = filepath[:-4] + '.txt'
            if os.path.exists(txt_path):
                self._add_to_pending(filepath, txt_path)
        elif filepath.endswith('.txt'):
            bin_path = filepath[:-4] + '.bin'
            if os.path.exists(bin_path):
                self._add_to_pending(bin_path, filepath)
    
    def _add_to_pending(self, bin_path, txt_path):
        """Add a file pair to pending transfers."""
        rel_path = os.path.relpath(bin_path, self.source_path)
        dir_path = os.path.dirname(rel_path)
        
        self.pending_files[bin_path] = {
            'txt_path': txt_path,
            'relative_dir': dir_path
        }
        
        logging.info(f"Added to pending transfers: {rel_path}")
        
        if len(self.pending_files) >= self.batch_size:
            self.process_pending_files()
    
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

def run_watcher():
    """Run the file watcher with enhanced error handling"""
    retry_delay = 60  # seconds
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            handler = DataTransferHandler()
            observer = Observer()
            observer.schedule(handler, handler.source_path, recursive=True)
            observer.start()
            
            logging.info(f"Started watching: {handler.source_path}")
            logging.info(f"Transferring to: {handler.hpc_experiment_path}")
            
            while True:
                time.sleep(1)
                handler.process_pending_files()
                
        except Exception as e:
            logging.error(f"Watcher error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached, exiting.")
                raise
        finally:
            try:
                observer.stop()
                observer.join()
            except Exception as e:
                logging.error(f"Error stopping observer: {str(e)}")

if __name__ == "__main__":
    run_watcher()