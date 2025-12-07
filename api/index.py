import os
import pickle
import zipfile
import gdown
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from preprocessing import TextPreprocessor

MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL")
MODEL_DIR = "saved_model"
MODEL_ZIP_PATH = "saved_model.zip"

# -----------------------------
# Download & Load Model
# -----------------------------
def ensure_model_downloaded():
    if os.path.exists(MODEL_DIR):
        return

    print("Downloading model from Google Drive...")
    gdown.download(MODEL_ZIP_URL, MODEL_ZIP_PATH, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as z:
        z.extractall(".")

    print("Model extracted successfully.")


model = None
vectorizer = None
svd = None
label_encoder = None
preprocessor = TextPreprocessor()


def load_models_once():
    global model, vectorizer, svd, label_encoder

    if model is not None:
        return

    ensure_model_downloaded()

    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf.pkl")
    svd_path = os.path.join(MODEL_DIR, "svd.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    model = load_pickle(model_path)
    vectorizer = load_pickle(vectorizer_path)

    if os.path.exists(svd_path):
        svd = load_pickle(svd_path)

    label_encoder = load_pickle(encoder_path)

    print("Model loaded successfully!")


def predict_text(text: str):
    load_models_once()

    processed = preprocessor.preprocess(text)
    X = vectorizer.transform([processed])

    if svd:
        X = svd.transform(X)

    pred = model.predict(X)[0]

    if isinstance(pred, str):
        label = pred
        confidence = None
    else:
        label = label_encoder.classes_[pred]
        try:
            confidence = float(model.predict_proba(X)[0][pred])
        except:
            confidence = None

    return {"label": label, "confidence": confidence}


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None, "confidence": None},
    )


@app.post("/", response_class=HTMLResponse)
async def index_post(request: Request):
    form = await request.form()
    text = form.get("text", "").strip()

    if not text:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": None, "confidence": None},
        )

    result = predict_text(text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result["label"],
            "confidence": result["confidence"],
        },
    )


@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    if "text" not in data:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        result = predict_text(data["text"])
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
