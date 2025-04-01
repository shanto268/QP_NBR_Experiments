import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

import globus_sdk
from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Load environment variables
load_dotenv()

class DataTransferHandler(FileSystemEventHandler):
    def __init__(self):
        self.source_path = os.getenv('LOCAL_STORAGE_PATH')
        self.destination_path = os.getenv('HPC_BASE_PATH')
        self.project_name = os.getenv('PROJECT_NAME')
        self.device_name = os.getenv('DEVICE_NAME')
        
        if not all([self.source_path, self.destination_path, 
                   self.project_name, self.device_name]):
            raise ValueError("Missing required environment variables")
            
        # Create the full HPC path
        self.hpc_experiment_path = os.path.join(
            self.destination_path,
            self.project_name,
            self.device_name
        )
        
        self.batch_size = os.getenv('GLOBUS_BATCH_SIZE')
        self.pending_files = {}
        self.transfer_client = self._setup_globus_client()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='data_transfer.log'
        )
        
    def _get_file_hash(self, filepath):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
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
    
    def process_pending_files(self):
        """Process pending file transfers."""
        if not self.pending_files:
            return
            
        try:
            transfer_data = globus_sdk.TransferData(
                self.transfer_client,
                source_endpoint=os.getenv('GLOBUS_SOURCE_ENDPOINT'),
                destination_endpoint=os.getenv('GLOBUS_DEST_ENDPOINT'),
                verify_checksum=True
            )
            
            files_to_transfer = self.pending_files.copy()
            
            # Add all pending files maintaining directory structure
            for bin_path, file_info in files_to_transfer.items():
                txt_path = file_info['txt_path']
                rel_dir = file_info['relative_dir']
                
                # Create destination directory structure under project/device
                dest_dir = os.path.join(self.hpc_experiment_path, rel_dir)
                
                # Add both .bin and .txt files to transfer
                for src_path in [bin_path, txt_path]:
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(dest_dir, filename)
                    
                    transfer_data.add_item(
                        source_path=src_path,
                        destination_path=dest_path,
                        recursive=False
                    )
                    logging.info(f"Adding to transfer: {src_path} -> {dest_path}")
            
            # Submit transfer
            task_id = self.transfer_client.submit_transfer(transfer_data)
            logging.info(f"Submitted transfer task: {task_id}")
            
            # Wait for completion
            while not self.transfer_client.task_wait(task_id, timeout=60):
                time.sleep(10)
                
            # Verify transfer
            task = self.transfer_client.get_task(task_id)
            if task['status'] == 'SUCCEEDED':
                # Verify files and clean up
                for bin_path, file_info in files_to_transfer.items():
                    txt_path = file_info['txt_path']
                    
                    # Remove files after successful transfer
                    os.remove(bin_path)
                    os.remove(txt_path)
                    del self.pending_files[bin_path]
                    
                    logging.info(f"Successfully transferred and removed: {bin_path}")
                    logging.info(f"Successfully transferred and removed: {txt_path}")
                    
                    # Clean up empty directories
                    dir_path = os.path.dirname(bin_path)
                    while dir_path != self.source_path:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            logging.info(f"Removed empty directory: {dir_path}")
                        dir_path = os.path.dirname(dir_path)
            else:
                logging.error(f"Transfer failed for task {task_id}, keeping local files")
                
        except Exception as e:
            logging.error(f"Transfer error: {str(e)}")

def run_watcher():
    handler = DataTransferHandler()
    observer = Observer()
    observer.schedule(handler, handler.source_path, recursive=True)
    observer.start()
    
    logging.info(f"Started watching: {handler.source_path}")
    logging.info(f"Transferring to: {handler.destination_path}")
    
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