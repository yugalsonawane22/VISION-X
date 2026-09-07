import cv2
import os
import insightface
import numpy as np

# Load the Buffalo model (same as Immich)
model = insightface.app.FaceAnalysis(name="buffalo_l", root="./.venv")
model.prepare(ctx_id=0)  # ctx_id=0 for CPU, use GPU id if available

# Path to peoples directory
PEOPLES_DIR = "peoples"

# Store embeddings and names
known_embeddings = []
known_names = []

# Load reference images
for filename in os.listdir(PEOPLES_DIR):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(PEOPLES_DIR, filename)
        img = cv2.imread(path)
        faces = model.get(img)
        if faces:
            # Take first face embedding
            embedding = faces[0].embedding
            known_embeddings.append(embedding)
            known_names.append(os.path.splitext(filename)[0])
print(f"Loaded {len(known_embeddings)} known faces.")

# Initialize webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = model.get(frame)
    for face in faces:
        embedding = face.embedding
        sims = [np.dot(embedding, ref) / (np.linalg.norm(embedding) * np.linalg.norm(ref))
                for ref in known_embeddings]
        best_match_idx = np.argmax(sims)
        best_score = sims[best_match_idx]
        if best_score > 0.35:
            name = known_names[best_match_idx]
        else:
            name = "Unknown"
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Detected: {name} (score={best_score:.2f})")
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
