# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import joblib
# import numpy as np

# app = FastAPI(title='Iris RF Demo')

# # Load or create a model
# try:
#     model = joblib.load('artifacts/rf_iris.joblib')
# except Exception:
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.datasets import load_iris
#     from sklearn.model_selection import train_test_split

#     iris = load_iris()
#     X_train, X_test, y_train, y_test = train_test_split(
#         iris.data, iris.target, test_size=0.2, random_state=42
#     )
#     model = RandomForestClassifier(n_estimators=10, random_state=42)
#     model.fit(X_train, y_train)

# # Create templates directory handler
# templates = Jinja2Templates(directory="templates")

# # API endpoint (for JSON requests)
# class PredictRequest(BaseModel):
#     data: list  # list of feature vectors or a single feature vector

# @app.post("/predict")
# def predict(req: PredictRequest):
#     data = np.array(req.data)
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
#     preds = model.predict(data).tolist()
#     return {"predictions": preds}

# # Web route (for HTML form)
# @app.get("/", response_class=HTMLResponse)
# def form_get(request: Request):
#     return templates.TemplateResponse("form.html", {"request": request, "result": None})

# @app.post("/", response_class=HTMLResponse)
# def form_post(
#     request: Request,
#     f1: float = Form(...),
#     f2: float = Form(...),
#     f3: float = Form(...),
#     f4: float = Form(...)
# ):
#     data = np.array([[f1, f2, f3, f4]])
#     preds = model.predict(data).tolist()
#     return templates.TemplateResponse("form.html", {"request": request, "result": preds[0]})




from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.testclient import TestClient
import joblib
import numpy as np

app = FastAPI(title='Iris RF Demo')

# Load or create a model
try:
    model = joblib.load('artifacts/rf_iris.joblib')
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

templates = Jinja2Templates(directory="templates")

class PredictRequest(BaseModel):
    data: list

@app.post("/predict")
def predict(req: PredictRequest):
    data = np.array(req.data)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # 🧩 Validation: Ensure each sample has 4 features
    if data.shape[1] != 4:
        return {"error": f"Expected 4 features per sample, got {data.shape[1]}"}

    preds = model.predict(data).tolist()
    return {"predictions": preds}


@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    f1: float = Form(...),
    f2: float = Form(...),
    f3: float = Form(...),
    f4: float = Form(...)
):
    # Build sample
    sample = [f1, f2, f3, f4]

    # Internal test client to hit /predict
    client = TestClient(app)
    resp = client.post("/predict", json={"data": sample})

    result = {
        "input": sample,
        "status_code": resp.status_code,
        "response_json": resp.json()
    }

    return templates.TemplateResponse("form.html", {"request": request, "result": result})
