import os
import sys
import io
import base64
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import utils

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# Update this if you want to use another checkpoint.
CHECKPOINT = PROJECT_ROOT / "!Results" / "11-10_18-08-56" / "model.pth"
DEVICE = torch.device("cpu")
IMAGE_SIZE = 544

CLASS_NAMES = [
    "Animal",
    "Masking background",
    "Non-masking background",
    "Foreground attention",
]
COLORS = np.asarray(config.COLORS, dtype=np.uint8)

TEAM_MEMBERS = [
    {"name": "SUBIYA ANJUM", "usn": "EDIT-USN-001", "role": "Model Development", "contribution": "Dataset preparation, model training, and segmentation experiments.", "initials": "SA", "photo": "image1.png"},
    {"name": "SANIYA TAHSEEN", "usn": "EDIT-USN-002", "role": "Backend & Integration", "contribution": "Flask API, model integration, and inference pipeline.", "initials": "ST", "photo": "image2.jpg"},
    {"name": "SONU", "usn": "EDIT-USN-003", "role": "Frontend & UX", "contribution": "Responsive interface, visualizations, and user experience.", "initials": "S", "photo": "image3.jpg"},
    {"name": "SUHANA", "usn": "EDIT-USN-004", "role": "Testing & Documentation", "contribution": "Testing, evaluation, documentation, and project validation.", "initials": "S", "photo": "image4.jpg"},
]

model_ready = False
model_error = None


def load_model():
    global model_ready, model_error
    try:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
        config.DEVICE = DEVICE
        utils.load_model(str(CHECKPOINT))
        config.model.to(DEVICE)
        config.model.eval()
        model_ready, model_error = True, None
    except Exception as exc:
        model_ready, model_error = False, str(exc)
        app.logger.exception("Could not load segmentation model")


def preprocess_image(image):
    """
    Baseline preprocessing. IMPORTANT: match this to the transforms used in
    data_model/dataset_class.py and utils/augmentation.py for valid predictions.
    If training uses letterboxing or a different encoder preprocessing function,
    change this function to match it exactly.
    """
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor.to(DEVICE)


def pil_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_segmentation(image):
    if not model_ready:
        raise RuntimeError(f"Model is unavailable: {model_error or 'checkpoint failed to load'}")

    original = image.convert("RGB")
    original_size = original.size
    tensor = preprocess_image(original)

    with torch.inference_mode():
        logits = config.model(tensor)
        class_ids = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Colorize at model resolution, then scale the visualization to original size.
    colored = Image.fromarray(COLORS[class_ids], mode="RGB").resize(
        original_size, Image.Resampling.NEAREST
    )
    overlay = Image.blend(original, colored, alpha=0.46)

    # Percentages are calculated at model output resolution.
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    stats = []
    for class_id, count in zip(unique_ids, counts):
        idx = int(class_id)
        if idx < len(CLASS_NAMES):
            color = tuple(int(c) for c in COLORS[idx])
            stats.append({
                "name": CLASS_NAMES[idx],
                "percent": round(float(count) * 100 / class_ids.size, 2),
                "color": "#%02x%02x%02x" % color,
            })

    return {
        "original": pil_data_url(original),
        "mask": pil_data_url(colored),
        "overlay": pil_data_url(overlay),
        "stats": stats,
        "width": original_size[0],
        "height": original_size[1],
    }


@app.context_processor
def inject_globals():
    return {"model_ready": model_ready, "model_error": model_error}


@app.get("/")
def home():
    return render_template("home.html", page="home")


@app.get("/about")
def about():
    return render_template("about.html", page="about")


@app.get("/team")
def team():
    return render_template("team.html", page="team", members=TEAM_MEMBERS)


@app.route("/detect", methods=["GET", "POST"])
def detect():
    result = None
    error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Choose an image before running detection."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                result = run_segmentation(image)
            except Exception as exc:
                app.logger.exception("Image segmentation failed")
                error = str(exc)
    return render_template("detect.html", page="detect", result=result, error=error)


@app.post("/api/predict")
def api_predict():
    if not model_ready:
        return jsonify({"error": f"Model unavailable: {model_error}"}), 503
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400
    try:
        image = Image.open(uploaded.stream).convert("RGB")
        return jsonify(run_segmentation(image))
    except Exception as exc:
        app.logger.exception("API prediction failed")
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_ready, "model_error": model_error})


load_model()

if __name__ == "__main__":
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Model ready: {model_ready}")
    if model_error:
        print(f"Model load error: {model_error}")
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
