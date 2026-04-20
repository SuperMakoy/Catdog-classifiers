from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

app = Flask(__name__)

model   = joblib.load("rf_model.pkl")
scaler  = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

IMG_SIZE = 224

# ── Feature extraction (copied from your Colab) ──────────────────────────────

def extract_color_features(img):
    features = []
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for channel in cv2.split(rgb):
        features.extend([np.mean(channel), np.std(channel), skew(channel.flatten())])
    for channel in cv2.split(hsv):
        features.extend([np.mean(channel), np.std(channel)])
    features.extend([np.mean(gray), np.std(gray),
                     skew(gray.flatten()), kurtosis(gray.flatten()),
                     shannon_entropy(gray)])
    hist = cv2.calcHist([rgb], [0], None, [16], [0, 256]).flatten()
    features.extend(hist)
    return features

def extract_glcm_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    return [graycoprops(glcm, p)[0, 0]
            for p in ('contrast', 'correlation', 'energy', 'homogeneity')]

def extract_lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    return list(hist) + [np.mean(lbp), np.std(lbp)]

def extract_hog_features(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), feature_vector=True)
    return list(hog_feat[:50]) + [np.mean(hog_feat), np.std(hog_feat)]

def extract_shape_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0] * 6
    c = max(contours, key=cv2.contourArea)
    area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    return [area, perimeter, w, h, w / h if h else 0, len(contours)]

def predict(image_bytes):
    arr  = np.frombuffer(image_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    feats  = (extract_color_features(img) + extract_glcm_features(img) +
              extract_lbp_features(img)   + extract_hog_features(img) +
              extract_shape_features(img))

    feats  = scaler.transform(np.array(feats).reshape(1, -1))
    feats  = selector.transform(feats)
    label  = model.predict(feats)[0]
    proba  = model.predict_proba(feats)[0]

    return {
        "label":      "Cat" if label == 0 else "Dog",
        "emoji":      "🐱"  if label == 0 else "🐶",
        "confidence": round(float(max(proba)) * 100, 1)
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    result = predict(request.files["file"].read())
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)