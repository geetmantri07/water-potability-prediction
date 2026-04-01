from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

# Import constants and pipelines
from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import WaterData, WaterDataClassifier
from src.pipeline.training_pipeline import TrainPipeline
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator


import time

# Initialize FastAPI
app = FastAPI()

Instrumentator().instrument(app).expose(app)

# -----------------------------
# PROMETHEUS METRICS
# -----------------------------
REQUEST_COUNT = Counter(
    "request_count_total",
    "Total HTTP requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["endpoint"]
)

# Static + Templates
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MIDDLEWARE FOR METRICS
# -----------------------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(duration)

    return response


# -----------------------------
# FORM CLASS (UPDATED)
# -----------------------------
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request

        self.ph: Optional[float] = None
        self.Hardness: Optional[float] = None
        self.Solids: Optional[float] = None
        self.Chloramines: Optional[float] = None
        self.Sulfate: Optional[float] = None
        self.Conductivity: Optional[float] = None
        self.Organic_carbon: Optional[float] = None
        self.Trihalomethanes: Optional[float] = None
        self.Turbidity: Optional[float] = None

    async def get_water_data(self):
        form = await self.request.form()

        self.ph = float(form.get("ph"))
        self.Hardness = float(form.get("Hardness"))
        self.Solids = float(form.get("Solids"))
        self.Chloramines = float(form.get("Chloramines"))
        self.Sulfate = float(form.get("Sulfate"))
        self.Conductivity = float(form.get("Conductivity"))
        self.Organic_carbon = float(form.get("Organic_carbon"))
        self.Trihalomethanes = float(form.get("Trihalomethanes"))
        self.Turbidity = float(form.get("Turbidity"))


# -----------------------------
# HOME PAGE
# -----------------------------
@app.get("/", tags=["Home"])
async def index(request: Request):
    return templates.TemplateResponse(
        "waterdata.html", {"request": request, "context": "Rendering"}
    )


# -----------------------------
# TRAIN MODEL
# -----------------------------
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


# -----------------------------
# PREDICTION
# -----------------------------
@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_water_data()

        water_data = WaterData(
            ph=form.ph,
            Hardness=form.Hardness,
            Solids=form.Solids,
            Chloramines=form.Chloramines,
            Sulfate=form.Sulfate,
            Conductivity=form.Conductivity,
            Organic_carbon=form.Organic_carbon,
            Trihalomethanes=form.Trihalomethanes,
            Turbidity=form.Turbidity
        )

        df = water_data.get_water_input_data_frame()

        model_predictor = WaterDataClassifier()
        prediction = model_predictor.predict(dataframe=df)[0]

        result = "Potable" if prediction == 1 else "Not Potable"

        return templates.TemplateResponse(
            "waterdata.html",
            {"request": request, "context": result},
        )

    except Exception as e:
        return {"status": False, "error": str(e)}
    
# -----------------------------
# METRICS ENDPOINT
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)