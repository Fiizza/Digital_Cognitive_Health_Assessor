import os
import pickle
import zipfile
import gdown
from flask import Flask, request, jsonify, render_template
from preprocessing import TextPreprocessor

MODEL_ZIP_URL = "https://drive.google.com/uc?id=1VghX-4I1tCCsOocOCB9jMMW_98a79GzM"

MODEL_DIR = "saved_model"
MODEL_ZIP_PATH = "saved_model.zip"

def ensure_model_downloaded():
    """Download and extract saved_model.zip if not already present."""
    if os.path.exists(MODEL_DIR):
        return  # Already downloaded

    print("Downloading model from Google Drive...")
    gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)

    print("Extracting saved_model.zip...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as z:
        z.extractall(".")

    print("Extraction complete.")



model = None
vectorizer = None
svd = None
label_encoder = None
preprocessor = TextPreprocessor()


def load_models_once():
    """Load ML artifacts only once per server invocation."""
    global model, vectorizer, svd, label_encoder

    if model is not None:
        return  # Already loaded

    ensure_model_downloaded()

    def load_pickle(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print("Loading model artifacts...")

    model = load_pickle(os.path.join(MODEL_DIR, "best_model.pkl"))
    vectorizer = load_pickle(os.path.join(MODEL_DIR, "tfidf.pkl"))

    svd_path = os.path.join(MODEL_DIR, "svd.pkl")
    if os.path.exists(svd_path):
        global svd
        svd = load_pickle(svd_path)

    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    label_encoder = load_pickle(label_encoder_path)

    print("All model artifacts loaded successfully.")


def predict_text(text):
    load_models_once()  # Ensure model is loaded

    processed = preprocessor.preprocess(text)
    X = vectorizer.transform([processed])

    if svd is not None:
        X = svd.transform(X)

    pred = model.predict(X)[0]

    if isinstance(pred, str):
        label = pred
        confidence = None
        try:
            proba = model.predict_proba(X)[0]
            idx = list(label_encoder.classes_).index(label)
            confidence = float(proba[idx])
        except:
            pass
    else:
        idx = int(pred)
        label = label_encoder.classes_[idx]
        try:
            confidence = float(model.predict_proba(X)[0][idx])
        except:
            confidence = None

    return {"label": label, "confidence": confidence}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            try:
                result = predict_text(text)
                prediction = result["label"]
                confidence = result["confidence"]
            except Exception as e:
                prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = predict_text(data["text"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
