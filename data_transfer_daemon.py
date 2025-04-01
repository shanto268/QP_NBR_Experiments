import logging
import os
import subprocess
import time
from pathlib import Path

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
        self.destination_path = os.getenv('HPC_BASE_PATH')
        self.project_name = os.getenv('PROJECT_NAME')
        self.device_name = os.getenv('DEVICE_NAME')
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        if not all([self.source_path, self.hpc_user, self.hpc_host, 
                   self.destination_path, self.project_name, self.device_name]):
            raise ValueError("Missing required environment variables")
            
        # Create the full HPC path
        self.hpc_experiment_path = os.path.join(
            self.destination_path,
            self.project_name,
            self.device_name
        )
        
        self.pending_files = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_transfer.log'
        )
        
        # Create remote directory structure
        self._ensure_remote_directory()
    
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
        import hashlib
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

    def _verify_transfer(self, local_path, remote_path):
        """Verify file transfer by comparing SHA-256 checksums"""
        local_checksum = self._get_file_checksum(local_path)
        remote_checksum = self._get_remote_checksum(remote_path)
        
        if local_checksum and remote_checksum:
            if local_checksum == remote_checksum:
                logging.info(f"Checksum verification successful for {local_path}")
                return True
            else:
                logging.error(f"Checksum mismatch for {local_path}")
                logging.error(f"Local: {local_checksum}")
                logging.error(f"Remote: {remote_checksum}")
                return False
        return False
    
    def process_pending_files(self):
        """Process pending file transfers using rsync."""
        if not self.pending_files:
            return
            
        files_to_transfer = self.pending_files.copy()
        
        for bin_path, file_info in files_to_transfer.items():
            txt_path = file_info['txt_path']
            rel_dir = file_info['relative_dir']
            
            # Create remote directory structure
            remote_dir = os.path.join(self.hpc_experiment_path, rel_dir)
            mkdir_cmd = f"ssh {self.hpc_user}@{self.hpc_host} 'mkdir -p {remote_dir}'"
            
            try:
                # Create remote directory
                subprocess.run(mkdir_cmd, shell=True, check=True)
                
                # Transfer both files
                for local_file in [bin_path, txt_path]:
                    filename = os.path.basename(local_file)
                    remote_path = f"{self.hpc_user}@{self.hpc_host}:{os.path.join(remote_dir, filename)}"
                    
                    # Use rsync for transfer
                    rsync_cmd = f"rsync -av --progress {local_file} {remote_path}"
                    subprocess.run(rsync_cmd, shell=True, check=True)
                    
                    # Verify transfer
                    remote_file = os.path.join(remote_dir, filename)
                    if self._verify_transfer(local_file, remote_file):
                        os.remove(local_file)
                        logging.info(f"Successfully transferred and removed: {local_file}")
                    else:
                        logging.error(f"Transfer verification failed for: {local_file}")
                        continue
                
                # Remove from pending files
                del self.pending_files[bin_path]
                
                # Clean up empty directories
                dir_path = os.path.dirname(bin_path)
                while dir_path != self.source_path:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logging.info(f"Removed empty directory: {dir_path}")
                    dir_path = os.path.dirname(dir_path)
                    
            except subprocess.CalledProcessError as e:
                logging.error(f"Transfer failed: {e}")

def run_watcher():
    handler = DataTransferHandler()
    observer = Observer()
    observer.schedule(handler, handler.source_path, recursive=True)
    observer.start()
    
    logging.info(f"Started watching: {handler.source_path}")
    logging.info(f"Transferring to: {handler.hpc_experiment_path}")
    
    try:
        while True:
            time.sleep(1)
            handler.process_pending_files()
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping file watcher")
    observer.join()

if __name__ == "__main__":
    run_watcher()