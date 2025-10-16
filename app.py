from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class PredictRequest(BaseModel):
    data: list  # list of feature vectors or a single feature vector

app = FastAPI(title='Iris RF Demo')

# For demo, load a model saved earlier if present; otherwise create a dummy model
try:
    model = joblib.load('artifacts/rf_iris.joblib')
except Exception as e:
    # fallback: a simple scikit-learn model trained quickly
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

@app.post('/predict')
def predict(req: PredictRequest):
    data = np.array(req.data)
    # ensure 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    preds = model.predict(data).tolist()
    return {'predictions': preds}