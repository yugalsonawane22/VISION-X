import base64
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests, json

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

app = Flask(__name__)
CORS(app)

known_embeddings = []
known_names = []

def load_images():
    global known_embeddings, known_names
    for filename in os.listdir(PEOPLES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(PEOPLES_DIR, filename)
            img = cv2.imread(path)
            faces = model.get(img)
            if faces:
                print(f"Found {len(faces)} face(s) in {filename}")
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(os.path.splitext(filename)[0])

@app.route("/")
def index():
    return render_template('UI2.html')

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "gemma3:27b-cloud"

SYSTEM_PROMPT = """You are VisionX, an AI assistant embedded in a smart camera system.
You receive an image and a list of recognized people detected by the face-recognition module.
Your job is to generate two things:
1. A brief, natural-language DESCRIPTION of the scene — who is present, what they might be doing, and any relevant context you can infer from the image. Keep it under 3 sentences.
2. A SHORT label (max 20 characters, no punctuation) suitable for a tiny 128×32 OLED display — think of it as a one-glance summary.

Always respond in the following strict JSON format (no markdown, no extra keys):
{"analysis": "<full description>", "analysis-short": "<≤20 char label>"}

### Few-shot examples ###

User: People in frame: ["Alice", "Bob"]. [image attached]
Assistant: {"analysis": "Alice and Bob are present in the frame, appearing to have a conversation in what looks like an office environment. Both are facing the camera and seem relaxed.", "analysis-short": "Alice & Bob"}

User: People in frame: ["Charlie"]. [image attached]
Assistant: {"analysis": "Charlie is the only recognized person in the scene. He appears to be working at a desk, focused on something off-camera.", "analysis-short": "Charlie seen"}

User: People in frame: []. 1 unknown person detected. [image attached]
Assistant: {"analysis": "One unidentified person is present in the frame. The individual's face was detected but did not match any known profiles.", "analysis-short": "1 Unknown"}

User: People in frame: ["Diana", "Eve"]. 2 unknown persons detected. [image attached]
Assistant: {"analysis": "Diana and Eve are recognized in the scene alongside 2 unidentified individuals. The group appears to be gathered in a common area.", "analysis-short": "Diana Eve +2"}
"""

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
    image_bytes = image_file.read()

    # Encode image as base64 for Ollama vision
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Call Ollama with gemma3:12b (vision-capable)
    # try:
    ollama_payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": "Describe what you see in this image.",
                "images": [image_b64]
            }
        ]
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=ollama_payload,
        timeout=60
    )
    resp.raise_for_status()
    print(f"[VisionX] Ollama response: {resp.text}")
    description = resp.json()["message"]["content"].strip()
    try:
        description_json = json.loads(description)
        description = description_json.get("analysis", "No description provided.")
        short_analysis = description_json.get("analysis-short", "No label provided.")
    except json.JSONDecodeError:
        # Fallback: if not JSON, use the response as description and generate short label
        description = description
        # Generate short label using the model
        short_payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": f"Summarize this description in 30 characters or less: {description[:500]}"  # Limit input to avoid too long
                }
            ]
        }
        try:
            short_resp = requests.post(f"{OLLAMA_URL}/api/chat", json=short_payload, timeout=30)
            short_resp.raise_for_status()
            short_analysis = short_resp.json()["message"]["content"].strip()[:30]
        except Exception:
            short_analysis = description[:30]
    print(f"[VisionX] {description}")

    # except Exception as e:
    #     print(f"[VisionX] Ollama error: {e}")
    #     description = "Failed to get description from model."
    #     short_analysis = "Error"

    return jsonify({
        "success": True,
        "result": {
            "filename": image_file.filename,
            "mime_type": image_file.mimetype or "image/jpeg",
            "size_bytes": len(image_bytes),
            "description": description,
            "short-description": short_analysis
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)