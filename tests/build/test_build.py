import os
import subprocess
import time
import requests
from dotenv import load_dotenv

class TestBuild:

    def test_streamlit_server_startup(self):
        """Test that Streamlit server can start and respond to requests"""
        # Load environment variables
        load_dotenv()
        test_port = os.getenv("TEST_PORT", "8502")
        
        # Get the project root (two levels up from tests/build/test_build.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Correct path to main.py in the src directory
        app_path = os.path.join(project_root, "src", "main.py")

        # Start Streamlit server in background
        process = subprocess.Popen(
            [
                "poetry",
                "run",
                "streamlit",
                "run",
                app_path,
                f"--server.port={test_port}",
                "--server.headless=true",
            ],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server to start (max 10 seconds)
            max_wait = 10
            wait_time = 0
            server_ready = False

            while wait_time < max_wait:
                try:
                    response = requests.get(f"http://localhost:{test_port}", timeout=2)
                    if response.status_code == 200:
                        server_ready = True
                        break
                except requests.RequestException:
                    pass

                time.sleep(1)
                wait_time += 1

            # If server didn't start, check what happened
            if not server_ready:
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    print(f"Process STDOUT: {stdout}")
                    print(f"Process STDERR: {stderr}")
                except subprocess.TimeoutExpired:
                    print("Process still running but not responding")
                    
            assert server_ready, f"Streamlit server failed to start within {max_wait} seconds"

            # Test that the server is actually serving the app
            response = requests.get(f"http://localhost:{test_port}", timeout=5)
            assert (
                response.status_code == 200
            ), f"Server responded with status {response.status_code}"

        finally:
            # Clean up: terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
