"""
run_app.py — Smart launcher for Streamlit Dashboard.
If the server is already running, immediately opens the browser.
If not, starts the engine and opens the browser.
"""
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

def is_server_alive(port=8501):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        # Fallback raw socket check
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                return s.connect_ex(('127.0.0.1', port)) == 0
        except Exception:
            return False

def main():
    project_root = Path(__file__).resolve().parent
    port = 8501
    url = f"http://localhost:{port}"

    print("=====================================================================")
    print("   PUMP-WATCH TERMINAL - AI Early Warning & Detection System")
    print("=====================================================================")

    if is_server_alive(port):
        print(f"[*] Radar Engine is ALREADY RUNNING on {url}!")
        print("[*] Opening your browser directly...")
        webbrowser.open(url)
        time.sleep(1.5)
        print("[*] Ready. You can close this window anytime.")
        sys.exit(0)

    # Isolate Streamlit configuration to local project folder
    local_streamlit_dir = str(project_root / ".streamlit")
    os.environ["STREAMLIT_CONFIG_DIR"] = local_streamlit_dir
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    print(f"[*] Starting Surveillance Engine on port {port}...")
    print(f"[*] Dashboard URL: {url}")
    print("[*] Opening browser in 2 seconds...")

    def open_browser():
        time.sleep(2.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=open_browser)
    t.daemon = True
    t.start()

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "app.py"),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
