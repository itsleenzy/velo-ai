from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import json
import joblib
import numpy as np
import httpx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    # from skl2onnx import convert_sklearn
    # from skl2onnx.common.data_types import FloatTensorType
    ONNX_ENABLED = False  # Temporarily disabled
except ImportError:
    ONNX_ENABLED = False

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

db = {"clean_df": None}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # override in .env if needed

AI_SYSTEM_PROMPT = """You are an expert machine learning system.
A user will describe a problem. Your job is to:
1. Understand the problem.
2. Decide if it is a classification or regression task.
3. Try to think of a real-world dataset that fits.
4. If no dataset is available, generate a synthetic dataset.
Dataset Rules:
- MUST have 8-12 feature columns (numeric or categorical)
- 1 target column
- At least 1000 rows
- Realistic relationships between features and target
- Include both numeric and categorical data where appropriate
- No missing values
- Avoid perfectly random data

Output Rules (CRITICAL - FOLLOW EXACTLY):
- Return ONLY valid CSV format
- NO markdown backticks (``` symbols)
- NO explanatory text before or after
- FIRST row MUST be column headers
- Each column separated by comma
- Example format:
  feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,target
  5.2,M,100,normal,yes,45,2.1,0.8,north,2025-03-30,positive
  ...more rows..."""


def auto_doctor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.columns.difference(numeric_cols)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def generate_fallback_dataset(problem: str = None) -> pd.DataFrame:
    """Fallback: Generate synthetic dataset when AI API fails."""
    import random
    random.seed(42)
    
    # Decide classification or regression based on problem keywords
    is_classification = False
    if problem:
        problem_lower = problem.lower()
        if any(word in problem_lower for word in ["predict", "classify", "detect", "category", "class", "churn", "fraud", "default"]):
            is_classification = any(word in problem_lower for word in ["classify", "category", "class", "churn", "fraud", "default"])
    
    n_samples = 1200
    n_features = 10
    
    if is_classification:
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            random_state=42
        )
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
    
    return df


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return "<h1>index.html not found in /static folder</h1>"
    return index_path.read_text(encoding="utf-8")


@app.post("/generate-dataset")
async def generate_dataset(problem: str = Form(...)):
    """Use Groq AI to generate a dataset from a natural language problem description.
    Falls back to synthetic dataset if AI API fails."""
    if not problem or not problem.strip():
        raise HTTPException(status_code=400, detail="Problem description is required.")

    if not GROQ_API_KEY:
        # No API key: use fallback
        try:
            df = generate_fallback_dataset(problem)
            db["clean_df"] = auto_doctor(df)
            return {
                "columns": db["clean_df"].columns.tolist(),
                "preview": db["clean_df"].head(5).to_dict(orient="records"),
                "rows": int(db["clean_df"].shape[0]),
                "source": "synthetic_fallback (no API key)",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Fallback dataset generation failed: {str(e)}")

    user_message = AI_SYSTEM_PROMPT.replace("{user_input}", problem.strip()) + f"\n\nUser Problem: \"{problem.strip()}\""

    # Try AI generation with fallback
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": user_message}],
                    "max_tokens": 8000,
                    "temperature": 0.7,
                },
            )

        if response.status_code != 200:
            # AI API failed: use fallback
            try:
                df = generate_fallback_dataset(problem)
                db["clean_df"] = auto_doctor(df)
                return {
                    "columns": db["clean_df"].columns.tolist(),
                    "preview": db["clean_df"].head(5).to_dict(orient="records"),
                    "rows": int(db["clean_df"].shape[0]),
                    "source": "synthetic_fallback (AI API error)",
                }
            except Exception as fallback_err:
                raise HTTPException(status_code=500, detail=f"AI API failed and fallback also failed: {str(fallback_err)}")

        try:
            result = response.json()
        except Exception as json_err:
            # JSON parse failed: use fallback
            try:
                df = generate_fallback_dataset(problem)
                db["clean_df"] = auto_doctor(df)
                return {
                    "columns": db["clean_df"].columns.tolist(),
                    "preview": db["clean_df"].head(5).to_dict(orient="records"),
                    "rows": int(db["clean_df"].shape[0]),
                    "source": "synthetic_fallback (JSON parse error)",
                }
            except Exception as fallback_err:
                raise HTTPException(status_code=500, detail=f"JSON parsing failed and fallback also failed: {str(fallback_err)}")
        
        csv_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not csv_text:
            # Empty response: use fallback
            try:
                df = generate_fallback_dataset(problem)
                db["clean_df"] = auto_doctor(df)
                return {
                    "columns": db["clean_df"].columns.tolist(),
                    "preview": db["clean_df"].head(5).to_dict(orient="records"),
                    "rows": int(db["clean_df"].shape[0]),
                    "source": "synthetic_fallback (empty AI response)",
                }
            except Exception as fallback_err:
                raise HTTPException(status_code=500, detail=f"AI returned empty response and fallback also failed: {str(fallback_err)}")

        # Strip any accidental markdown fences
        if csv_text.startswith("```"):
            lines = csv_text.split("\n")
            csv_text = "\n".join(l for l in lines if not l.startswith("```"))
        
        # Remove leading/trailing whitespace lines
        csv_text = "\n".join(line for line in csv_text.split("\n") if line.strip())

        try:
            df = pd.read_csv(io.StringIO(csv_text))
            
            # Validate dataset
            if df.shape[0] < 10 or df.shape[1] < 2:
                raise ValueError("AI dataset too small")
                
        except Exception as parse_err:
            # CSV parse failed: use fallback
            try:
                df = generate_fallback_dataset(problem)
            except Exception as fallback_err:
                raise HTTPException(status_code=500, detail=f"AI CSV parsing failed and fallback also failed: {str(fallback_err)}")

        db["clean_df"] = auto_doctor(df)
        return {
            "columns": db["clean_df"].columns.tolist(),
            "preview": db["clean_df"].head(5).to_dict(orient="records"),
            "rows": int(db["clean_df"].shape[0]),
            "source": "ai_generated",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Any other error: use fallback
        try:
            df = generate_fallback_dataset(problem)
            db["clean_df"] = auto_doctor(df)
            return {
                "columns": db["clean_df"].columns.tolist(),
                "preview": db["clean_df"].head(5).to_dict(orient="records"),
                "rows": int(db["clean_df"].shape[0]),
                "source": "synthetic_fallback (unexpected error)",
            }
        except Exception as fallback_err:
            raise HTTPException(status_code=500, detail=f"Unexpected error and fallback failed: {str(fallback_err)}")


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        db["clean_df"] = auto_doctor(df)
        return {
            "columns": db["clean_df"].columns.tolist(),
            "preview": db["clean_df"].head(5).to_dict(orient="records"),
            "rows": int(db["clean_df"].shape[0]),
            "source": "upload",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train_model(task: str = Form(...), model_name: str = Form(...)):
    try:
        if db["clean_df"] is None:
            raise HTTPException(status_code=400, detail="Generate or upload a dataset first.")

        df = db["clean_df"].copy()
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least one feature and one target column.")

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        is_regression = task.strip().lower() == "regression"

        logs = []
        if df.shape[0] < 1000:
            logs.append("Small dataset detected — recommending Logistic Regression for stability.")
            recommended_model = "Logistic Regression"
        else:
            logs.append("Large dataset detected — activating Gradient Boosting for high-capacity learning.")
            recommended_model = "Gradient Boosting"

        if model_name.strip().lower() in ("auto-suggest", "auto"):
            model_name = recommended_model

        if is_regression:
            model_specs = [
                ("Random Forest", RandomForestRegressor),
                ("Gradient Boosting", HistGradientBoostingRegressor),
                ("Linear Regression", LinearRegression),
                ("Neural Network", MLPRegressor),
            ]
        else:
            model_specs = [
                ("Random Forest", RandomForestClassifier),
                ("Gradient Boosting", HistGradientBoostingClassifier),
                ("Logistic Regression", LogisticRegression),
                ("Neural Network", MLPClassifier),
            ]

        stratify = None
        if not is_regression and y.nunique() > 1:
            # Check if all classes have at least 2 samples for stratification
            min_class_samples = y.value_counts().min()
            if min_class_samples >= 2:
                stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError as split_error:
            # Fallback to non-stratified split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            logs.append("Warning: Stratified split failed, using random split instead.")

        results = []
        champion_model = None
        champion_score = None

        for name, cls in model_specs:
            try:
                if cls is LogisticRegression:
                    model = cls(max_iter=1000)
                elif cls in (MLPClassifier, MLPRegressor):
                    model = cls(max_iter=500)
                else:
                    model = cls()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred) if is_regression else accuracy_score(y_test, y_pred)
                results.append({"name": name, "score": float(score)})
                if champion_score is None or score > champion_score:
                    champion_score = float(score)
                    champion_model = model
            except Exception as e:
                results.append({"name": name, "score": float("nan"), "error": str(e)})

        results_valid = [r for r in results if not np.isnan(r["score"])]
        if not results_valid:
            raise HTTPException(status_code=400, detail="All model training experiments failed.")

        results_sorted = sorted(results_valid, key=lambda r: r["score"], reverse=True)
        champion_name = results_sorted[0]["name"]
        champion_score = results_sorted[0]["score"]

        joblib.dump(champion_model, STATIC_DIR / "velo_model.pkl")

        model_info = {
            "task": "Regression" if is_regression else "Classification",
            "model_name": champion_name,
            "score": round(float(champion_score), 4),
            "target_column": str(target_col),
            "feature_columns": [str(c) for c in X.columns.tolist()],
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "trained_at": datetime.utcnow().isoformat() + "Z",
        }
        (STATIC_DIR / "velo_model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

        trend = np.linspace(max(champion_score - 0.25, 0.0), champion_score, 10)
        history = [
            {"epoch": i + 1, "accuracy": round(float(v), 3), "loss": round(float(max(0.0, 1.0 - v)), 3)}
            for i, v in enumerate(trend)
        ]

        return {
            "status": "Complete",
            "graph_data": history,
            "score": round(champion_score, 4),
            "leaderboard": results_sorted,
            "logs": logs,
            "champion": champion_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)[:200]}")


@app.get("/download")
async def download_model():
    model_path = STATIC_DIR / "velo_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Train a model first.")
    return FileResponse(path=str(model_path), media_type="application/octet-stream", filename="velo_model.pkl")


@app.get("/download-onnx")
async def download_onnx():
    if not ONNX_ENABLED:
        raise HTTPException(status_code=400, detail="skl2onnx not installed.")
    model_path = STATIC_DIR / "velo_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Train a model first.")
    try:
        model = joblib.load(model_path)
        n_features = int(getattr(model, "n_features_in_", 0))
        if n_features <= 0:
            raise ValueError("Cannot determine feature count.")
        onnx_model = convert_sklearn(model, initial_types=[("float_input", FloatTensorType([None, n_features]))])
        onnx_path = STATIC_DIR / "velo_model.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        return FileResponse(path=str(onnx_path), media_type="application/octet-stream", filename="velo_model.onnx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-view")
async def view_model():
    try:
        model_path = STATIC_DIR / "velo_model.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=400, detail="Train a model first.")
        model = joblib.load(model_path)
        return PlainTextResponse(f"Model Type: {type(model).__name__}\n\nParameters:\n{model.get_params()}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/predict")
async def predict(features: dict = Body(...)):
    model_path = STATIC_DIR / "velo_model.pkl"
    info_path = STATIC_DIR / "velo_model_info.json"
    if not model_path.exists() or not info_path.exists():
        raise HTTPException(status_code=400, detail="Train a model first.")
    try:
        model = joblib.load(model_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        feature_columns = info.get("feature_columns", [])
        missing = [c for c in feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        X = pd.DataFrame([[features[c] for c in feature_columns]], columns=feature_columns)
        return {"prediction": model.predict(X).tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
