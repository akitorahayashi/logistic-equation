import os
import subprocess
import time
import requests

class TestBuild:

    def test_streamlit_server_startup(self):
        """Test that Streamlit server can start and respond to requests"""
        # Correct path to main.py in the src directory
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "main.py",
        )

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Start Streamlit server in background
        process = subprocess.Popen(
            [
                "poetry",
                "run",
                "streamlit",
                "run",
                app_path,
                "--server.port=8502",
                "--server.headless=true",
            ],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server to start (max 15 seconds)
            max_wait = 15
            wait_time = 0
            server_ready = False

            while wait_time < max_wait:
                try:
                    response = requests.get("http://localhost:8502", timeout=2)
                    if response.status_code == 200:
                        server_ready = True
                        break
                except requests.RequestException:
                    pass

                time.sleep(1)
                wait_time += 1

            assert server_ready, "Streamlit server failed to start within 15 seconds"

            # Test that the server is actually serving the app
            response = requests.get("http://localhost:8502", timeout=5)
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
