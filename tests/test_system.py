import unittest
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import os
import signal
import sys
import socket
import platform
import contextlib
import threading

class PortfolioSystemTest(unittest.TestCase):
    @classmethod
    def wait_for_server(cls, timeout=30):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                if cls.flask_process.poll() is not None:
                    print("Server process terminated before ready.")
                    return False
                    
                # Try to connect
                response = requests.get(f"{cls.base_url}/")
                if response.status_code == 200:
                    return True
                print(f"Server responded with status code: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("Waiting for server to start...")
                time.sleep(1)
            except Exception as e:
                print(f"Error while checking server: {e}")
                time.sleep(1)
        return False

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8080"
        
        # Kill any existing process on port 8080
        try:
            kill_cmd = "lsof -t -i:8080 -sTCP:LISTEN | xargs -r kill -9"
            subprocess.run(kill_cmd, shell=True, capture_output=True)
        except Exception:
            pass
        
        # Start Flask server with environment variables
        env = os.environ.copy()
        env["FLASK_APP"] = "webapp.py"
        env["FLASK_ENV"] = "development"
        env["PYTHONUNBUFFERED"] = "1"  # Ensures Python output isn't buffered
        
        # Start server with output capture (single approach)
        cmd = ["python", "-m", "flask", "run", "--host", "127.0.0.1", "--port", "8080", "--no-reload"]
        cls.flask_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True
        )

        # Start background threads to continuously read output
        def log_output(stream, prefix):
            for line in stream:
                print(f"{prefix}: {line}", end="")

        stdout_thread = threading.Thread(target=log_output, args=(cls.flask_process.stdout, "SERVER OUT"))
        stderr_thread = threading.Thread(target=log_output, args=(cls.flask_process.stderr, "SERVER ERR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Wait for server to be ready
        if not cls.wait_for_server():
            raise Exception("Server failed to start within timeout")

        # Setup Selenium with retry mechanism
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        retry_count = 3
        for i in range(retry_count):
            try:
                cls.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                break
            except Exception as e:
                if i == retry_count - 1:
                    raise e
                time.sleep(2)

    def test_01_server_is_running(self):
        """Test if server is running and responding"""
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["Content-Type"])

    def test_02_stock_data_api(self):
        """Test if stock data API is working"""
        response = requests.post(f"{self.base_url}/analyze", json={"symbol": "RELIANCE.NS"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("price", data)
        self.assertIn("volume", data)
        self.assertIn("indicators", data)

    def test_03_frontend_elements(self):
        """Test if all frontend elements are present and visible"""
        self.driver.get(f"{self.base_url}/")
        
        # Test navigation elements
        self.assertTrue(self.is_element_present("top-bar"))
        self.assertTrue(self.is_element_present("portfolio-summary"))
        self.assertTrue(self.is_element_present("technical-indicators"))
        
        # Test sidebar
        self.assertTrue(self.is_element_present("toc-sidebar"))
        nav_links = self.driver.find_elements(By.CLASS_NAME, "toc-link")
        self.assertGreater(len(nav_links), 0)

    def test_04_stock_card_functionality(self):
        """Test if stock cards are loading and displaying data correctly"""
        self.driver.get(f"{self.base_url}/")
        
        # Wait for stock cards to load
        cards = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "stock-card"))
        )
        self.assertGreater(len(cards), 0)
        
        # Test first card elements
        first_card = cards[0]
        self.assertTrue(first_card.find_element(By.CLASS_NAME, "price").is_displayed())
        self.assertTrue(first_card.find_element(By.CLASS_NAME, "indicator-card").is_displayed())

    def test_05_technical_indicators(self):
        """Test if technical indicators are calculated and displayed correctly"""
        self.driver.get(f"{self.base_url}/")
        
        # Wait for indicators to load
        indicators = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "indicator-card"))
        )
        
        # Verify all required indicators are present
        indicator_types = ["RSI", "MACD", "Bollinger", "SMA", "Volume", "R/R"]
        for indicator in indicator_types:
            elements = self.driver.find_elements(By.XPATH, f"//span[contains(text(), '{indicator}')]")
            self.assertGreater(len(elements), 0, f"Missing indicator: {indicator}")

    def test_06_responsive_layout(self):
        """Test if the layout is responsive"""
        self.driver.get(f"{self.base_url}/")
        
        # Test different viewport sizes
        viewports = [
            (1920, 1080),  # Desktop
            (1024, 768),   # Tablet
            (375, 812)     # Mobile
        ]
        
        for width, height in viewports:
            self.driver.set_window_size(width, height)
            time.sleep(1)  # Allow time for layout adjustments
            
            # Verify key elements are still visible
            self.assertTrue(self.is_element_present("top-bar"))
            self.assertTrue(self.is_element_present("main-content"))

    def is_element_present(self, element_id):
        """Helper method to check if element is present"""
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, element_id))
            )
            return True
        except TimeoutException:
            return False

    @classmethod
    def tearDownClass(cls):
        # Clean up Selenium
        if hasattr(cls, 'driver'):
            try:
                cls.driver.quit()
            except Exception as e:
                print(f"Error closing Selenium driver: {e}")
        
        # Clean up Flask server
        if hasattr(cls, 'flask_process'):
            try:
                # Try graceful shutdown first
                cls.flask_process.terminate()
                try:
                    cls.flask_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Server didn't shut down gracefully, forcing...")
                    cls.flask_process.kill()  # Force kill if it doesn't respond
                    cls.flask_process.wait(timeout=5)
                
                # Get any final output
                try:
                    out, err = cls.flask_process.communicate(timeout=2)
                    if out:
                        print(f"Final server output:\n{out}")
                    if err:
                        print(f"Final server errors:\n{err}")
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"Error shutting down Flask server: {e}")
        
        # Clean up any remaining processes (platform independent)
        try:
            if sys.platform == 'win32':
                subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq flask"], 
                             capture_output=True)
            else:
                # Kill any process listening on our port
                subprocess.run(["lsof", "-t", "-i:8080", "-sTCP:LISTEN"], 
                             stdout=subprocess.PIPE,
                             text=True)
                kill_cmd = "lsof -t -i:8080 -sTCP:LISTEN | xargs -r kill -9"
                subprocess.run(kill_cmd, shell=True, capture_output=True)
        except Exception as e:
            print(f"Cleanup error: {e}")
            # Ensure process is killed even if other cleanup fails
            try:
                os.kill(cls.flask_process.pid, signal.SIGKILL)
            except:
                pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
