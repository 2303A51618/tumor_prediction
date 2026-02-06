import { useMemo, useState } from "react";
import "./App.css";

const rawApiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const API_URL = rawApiUrl.endsWith("/predict")
  ? rawApiUrl
  : `${rawApiUrl.replace(/\/$/, "")}/predict`;

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [isValidMri, setIsValidMri] = useState(true);

  const fileName = useMemo(() => (file ? file.name : "No file selected"), [file]);

  const validateMriLikeImage = (selectedFile) =>
    new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const targetSize = 128;
        canvas.width = targetSize;
        canvas.height = targetSize;
        ctx.drawImage(img, 0, 0, targetSize, targetSize);
        const { data } = ctx.getImageData(0, 0, targetSize, targetSize);

        let diffSum = 0;
        let samples = 0;
        for (let i = 0; i < data.length; i += 16) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          diffSum += Math.abs(r - g) + Math.abs(r - b) + Math.abs(g - b);
          samples += 1;
        }

        const avgDiff = diffSum / samples;
        resolve(avgDiff < 18);
      };
      img.onerror = () => resolve(false);
      img.src = URL.createObjectURL(selectedFile);
    });

  const handleFileChange = async (event) => {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError("");
    setIsValidMri(true);

    if (selected) {
      const url = URL.createObjectURL(selected);
      setPreviewUrl(url);
      const looksLikeMri = await validateMriLikeImage(selected);
      setIsValidMri(looksLikeMri);
      if (!looksLikeMri) {
        setError("Invalid MRI image. Please try again.");
      }
    } else {
      setPreviewUrl("");
    }
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a JPG or PNG image.");
      return;
    }

    if (!isValidMri) {
      setError("Invalid MRI image. Please try again.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err?.message || "Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <main className="card">
        <header className="header">
          <h1>Brain Tumor Detection</h1>
          <p>Upload an MRI image and get a prediction from the model.</p>
        </header>

        <section className="uploader">
          <label className="file-input">
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleFileChange}
            />
            <span>Choose image</span>
          </label>
          <span className="file-name">{fileName}</span>
        </section>

        {previewUrl && (
          <section className="preview">
            <img src={previewUrl} alt="Selected preview" />
          </section>
        )}

        <button className="predict-btn" onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <section className="results">
            <h2>Prediction</h2>
            <div className="result-row">
              <span className="label">Predicted class</span>
              <span className="value">{result.predicted_class}</span>
            </div>
            <div className="result-row">
              <span className="label">Confidence</span>
              <span className="value">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>

            <div className="probabilities">
              <h3>All class probabilities</h3>
              <ul>
                {Object.entries(result.all_probabilities || {}).map(
                  ([label, prob]) => (
                    <li key={label}>
                      <span>{label}</span>
                      <span>{(prob * 100).toFixed(2)}%</span>
                    </li>
                  )
                )}
              </ul>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
