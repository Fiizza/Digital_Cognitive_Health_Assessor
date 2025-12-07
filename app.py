import os
import pickle
from flask import Flask, request, jsonify, render_template

from preprocessing import TextPreprocessor

MODEL_DIR = "saved_model"

# --- load artifacts (fail with descriptive errors) ---
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("saved_model folder not found. Please run: python src/train_model.py")

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}. Re-run training (python src/train_model.py).")
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pickle(os.path.join(MODEL_DIR, "best_model.pkl"))
vectorizer = load_pickle(os.path.join(MODEL_DIR, "tfidf.pkl"))
svd = None
svd_path = os.path.join(MODEL_DIR, "svd.pkl")
if os.path.exists(svd_path):
    svd = load_pickle(svd_path)
label_encoder = load_pickle(os.path.join(MODEL_DIR, "label_encoder.pkl"))

preprocessor = TextPreprocessor()

# --- DIMENSION CHECKS ---
def get_vectorizer_output_dim(vect):
    # modern sklearn: get_feature_names_out
    try:
        return len(vect.get_feature_names_out())
    except Exception:
        # fallback: vocabulary_ dict length or max_features attr
        if hasattr(vect, "vocabulary_") and vect.vocabulary_:
            return len(vect.vocabulary_)
        return getattr(vect, "max_features", None)

vector_dim = get_vectorizer_output_dim(vectorizer)

svd_in_dim = None
svd_out_dim = None
if svd is not None:
    try:
        svd_in_dim = svd.components_.shape[1]
        svd_out_dim = svd.components_.shape[0]
    except Exception:
        svd_in_dim = None
        svd_out_dim = None

# Try to infer model expected dim more robustly
model_expected_dim = None
try:
    # linear models (coef_) shape
    model_expected_dim = model.coef_.shape[1]
except Exception:
    # try n_features_in_ (newer sklearn attribute)
    model_expected_dim = getattr(model, "n_features_in_", None)
    # if wrapped in a Pipeline, attempt to inspect final estimator
    if model_expected_dim is None and hasattr(model, "steps"):
        try:
            final = model.steps[-1][1]
            model_expected_dim = getattr(final, "coef_", None)
            if hasattr(final, "coef_"):
                model_expected_dim = final.coef_.shape[1]
            else:
                model_expected_dim = getattr(final, "n_features_in_", None)
        except Exception:
            model_expected_dim = None

mismatch_message = None
if model_expected_dim is not None:
    if svd is None and vector_dim is not None and model_expected_dim != vector_dim:
        mismatch_message = (
            f"Model expects {model_expected_dim} features but vectorizer produces {vector_dim} features, "
            "and no SVD was found. Recreate saved_model by running: python src/train_model.py"
        )
    elif svd is not None:
        if svd_out_dim is not None and svd_out_dim != model_expected_dim:
            mismatch_message = (
                f"Model expects {model_expected_dim} dims, but SVD produces {svd_out_dim} dims.\n"
                "Saved artifacts are mismatched. Delete saved_model and re-run training: python src/train_model.py"
            )
        if svd_in_dim is not None and vector_dim is not None and svd_in_dim != vector_dim:
            mismatch_message = (
                f"SVD expects input dim {svd_in_dim} but vectorizer has {vector_dim} features.\n"
                "Saved vectorizer and SVD do not match. Delete saved_model and re-run training: python src/train_model.py"
            )

if mismatch_message:
    print("ARTIFACT MISMATCH DETECTED:")
    print(mismatch_message)
    print("Diagnostic values:")
    print(f"  vectorizer features: {vector_dim}")
    print(f"  svd in dim: {svd_in_dim}")
    print(f"  svd out dim: {svd_out_dim}")
    print(f"  model expected dim: {model_expected_dim}")
    raise RuntimeError("Saved model artifacts mismatch. See message above.")

print("Loaded model artifacts — shapes look consistent (or model expected dim unknown).")

# --- predict function ---
def predict_text(text):
    processed = preprocessor.preprocess(text)
    X = vectorizer.transform([processed])

    if svd is not None:
        try:
            X = svd.transform(X)
        except Exception as e:
            # unexpected, but fail gracefully
            raise RuntimeError(f"SVD transform failed: {e}")

    # prediction
    pred = model.predict(X)[0]

    # if model outputs label strings (not encoded ints), handle both cases
    if isinstance(pred, (str, bytes)):
        label = pred if isinstance(pred, str) else pred.decode("utf-8")
        confidence = None
        # attempt to find index for probability if possible
        try:
            proba = model.predict_proba(X)[0]
            # find index of label in label_encoder classes
            if hasattr(label_encoder, "classes_"):
                idx = list(label_encoder.classes_).index(label)
                confidence = float(proba[idx])
        except Exception:
            confidence = None
    else:
        # model returns encoded index (int)
        pred_idx = int(pred)
        label = label_encoder.classes_[pred_idx]
        try:
            confidence = float(model.predict_proba(X)[0][pred_idx])
        except Exception:
            confidence = None

    return {"label": label, "confidence": confidence}

# --- flask app ---
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        t = request.form.get("text","").strip()
        if t:
            try:
                res = predict_text(t)
                prediction = res["label"]
                confidence = res["confidence"]
            except Exception as e:
                # user-friendly message
                return render_template("index.html", prediction=f"Prediction failed: {e}", confidence=None)
    return render_template("index.html", prediction=prediction, confidence=confidence)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error":"No text provided"}), 400
    try:
        return jsonify(predict_text(data["text"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # use 0.0.0.0 for external access and configurable PORT for cloud hosts
    import os
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)
