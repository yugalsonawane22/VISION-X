import os
import socket
import subprocess
import threading
import requests as req
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Allow override via env var; default to localhost where app.py runs
APP_PY_URL = os.environ.get("APP_PY_URL", "http://127.0.0.1:80/api/detect")
print(f"[termux] Forwarding detects → {APP_PY_URL}")
app = Flask(__name__)
CORS(app)  # Allow browser (visionx2.html) to POST /detect

def find_esp32(port=9087, timeout=0.2, max_workers=100):
    found_device = None
    def get_local_network_prefix():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # doesn't actually send data
            local_ip = s.getsockname()[0]
        finally:
            s.close()
        return ".".join(local_ip.split(".")[:3]) + "."

    def scan_ip(ip):
        nonlocal found_device
        if found_device:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        if result == 0 and not found_device:
            found_device = ip
        sock.close()
    network_prefix = get_local_network_prefix()
    ips = [network_prefix + str(i) for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(scan_ip, ips)
    return found_device

ESP_IP = find_esp32()

@app.route("/")
def index():
    if ESP_IP is None:
        return "ESP32 not found<a href='/scan'>Click here to Scan</a>"
    return render_template('visionx2.html', esp_ip=ESP_IP)

@app.route("/scan")
def scan():
    global ESP_IP
    ESP_IP = find_esp32()
    return jsonify({"esp_ip": ESP_IP})

@app.route("/health")
def health():
    """Quick reachability check — open this in a browser to confirm termux.py is up."""
    return jsonify({"status": "ok", "app_py_url": APP_PY_URL, "esp_ip": ESP_IP})

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "No image provided", 400
    image_file = request.files["image"]
    image_bytes = image_file.read()
    print(f"[termux] image received: {len(image_bytes)} bytes")

    # ── 2. Forward image to app.py /api/detect ─────────────────────────────
    try:
        print(f"[termux] POSTing to {APP_PY_URL} …")
        upstream = req.post(
            APP_PY_URL,
            files={"image": (image_file.filename or "frame.jpg", image_bytes, image_file.mimetype or "image/jpeg")},
            timeout=90
        )
        upstream.raise_for_status()
        data = upstream.json()
        print(f"[termux] app.py response: {data}")
    except req.exceptions.ConnectionError:
        msg = f"Cannot reach app.py at {APP_PY_URL} — is it running?"
        print(f"[termux] ERROR: {msg}")
        return msg, 502
    except req.exceptions.Timeout:
        msg = f"app.py timed out (90 s)"
        print(f"[termux] ERROR: {msg}")
        return msg, 504
    except Exception as e:
        msg = f"app.py error: {e}"
        print(f"[termux] ERROR: {msg}")
        return msg, 502

    result         = data.get("result", {})
    analysis       = result.get("description", "No description available.")
    # Since app.py no longer returns a short analysis, we generate a fallback for the OLED
    short_analysis = "Detected" 
    print(f"[termux] analysis='{analysis[:50]}...' short='{short_analysis}'")

    # ── 3. Push short label to ESP OLED ────────────────────────────────────
    if ESP_IP and short_analysis:
        try:
            req.get(
                f"http://{ESP_IP}/display",
                params={"msg": short_analysis},
                timeout=3
            )
            print(f"[termux] OLED updated: '{short_analysis}'")
        except Exception as e:
            print(f"[termux] ESP display error: {e}")

    # ── 4. Speak analysis via Termux TTS (non-blocking) ────────────────────
    def speak(text):
        try:
            subprocess.run(["termux-tts-speak", text], timeout=60)
        except FileNotFoundError:
            print("[termux] termux-tts-speak not found (not running in Termux?)")
        except Exception as e:
            print(f"[termux] TTS error: {e}")

    threading.Thread(target=speak, args=(analysis,), daemon=True).start()

    # ── 5. Return full analysis text to visionx2.html ──────────────────────
    return analysis, 200

os.system("termux-open http://localhost:5000/")  # Open the browser immediately
if __name__=="__main__":
    app.run(debug=False, host="0.0.0.0")