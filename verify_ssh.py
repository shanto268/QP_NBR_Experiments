import logging
import os
import subprocess

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def verify_ssh_setup():
    """Verify SSH setup and provide guidance for fixes"""
    load_dotenv()
    
    hpc_user = os.getenv('HPC_USERNAME')
    hpc_host = os.getenv('HPC_HOSTNAME')
    
    if not all([hpc_user, hpc_host]):
        logging.error("Missing HPC credentials in .env file")
        return False
        
    # Check SSH key existence
    ssh_key = os.path.expanduser('~/.ssh/id_rsa')
    if not os.path.exists(ssh_key):
        logging.error("No SSH key found. Please generate one using:")
        logging.error("ssh-keygen -t rsa -b 4096")
        return False
        
    # Test SSH connection
    test_cmd = f"ssh -v {hpc_user}@{hpc_host} echo 'test'"
    try:
        result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logging.info("SSH connection successful!")
            return True
            
        logging.error("SSH connection failed. Details:")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
        
        # Provide specific guidance based on error
        if "Permission denied" in result.stderr:
            logging.error("\nTo fix SSH key authentication:")
            logging.error("1. Copy your SSH key to HPC:")
            logging.error(f"   ssh-copy-id {hpc_user}@{hpc_host}")
            logging.error("2. Or manually add your public key to ~/.ssh/authorized_keys on HPC")
            logging.error("3. Check permissions:")
            logging.error("   chmod 700 ~/.ssh")
            logging.error("   chmod 600 ~/.ssh/authorized_keys")
            
        elif "Could not resolve hostname" in result.stderr:
            logging.error("\nHostname resolution failed:")
            logging.error("1. Check your .env file HPC_HOSTNAME value")
            logging.error("2. Verify you can ping the HPC host")
            logging.error("3. Check your VPN connection if required")
            
        elif "Connection refused" in result.stderr:
            logging.error("\nConnection refused:")
            logging.error("1. Verify the HPC system is up")
            logging.error("2. Check if you need to use a jump host")
            logging.error("3. Verify SSH service is running on HPC")
            
    except subprocess.TimeoutExpired:
        logging.error("Connection timed out. Check network connectivity")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return False
        
    return False

if __name__ == "__main__":
    print("Verifying SSH setup...")
    verify_ssh_setup() 