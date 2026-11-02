from pathlib import Path
import logging

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Initialize FastAPI and API documentation
# --------------------------------------------------

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="""
    ## Employee Prediction Service

    This API uses a trained Random Forest model to predict
    the target class for an employee record.

    ### Features
    - Validates incoming employee data.
    - Returns the predicted class.
    - Returns estimated probabilities for each class.
    - Provides health-check and prediction endpoints.

    **Note:** Confirm the dataset's target-label meanings
    before interpreting a predicted class as employee attrition.
    """,
    version="1.1.0",
    contact={
        "name": "ML API Support"
    }
)


# --------------------------------------------------
# 3. Load the saved model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_v1.joblib"

FEATURES = [
    "TotalMonthsOfExp",
    "TotalOrgsWorked",
    "MonthsInOrg",
    "LastPayIncrementBand",
    "AverageFeedback"
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

logger.info("Machine learning model loaded successfully")


# --------------------------------------------------
# 4. Request model
# --------------------------------------------------

class EmployeeData(BaseModel):
    TotalMonthsOfExp: float = Field(
        ge=0,
        description="Total professional experience in months",
        examples=[60]
    )

    TotalOrgsWorked: int = Field(
        ge=0,
        description="Total number of organizations worked at",
        examples=[3]
    )

    MonthsInOrg: float = Field(
        ge=0,
        description="Number of months in the current organization",
        examples=[24]
    )

    LastPayIncrementBand: float = Field(
        ge=0,
        description="Pay increment band encoded in the dataset",
        examples=[2]
    )

    AverageFeedback: float = Field(
        ge=0,
        description="Average feedback value encoded in the dataset",
        examples=[3]
    )


# --------------------------------------------------
# 5. Response models
# --------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_class: int = Field(
        description="Predicted target class"
    )

    class_probabilities: dict[str, float] = Field(
        description="Estimated probability for each target class"
    )

    model_version: str = Field(
        description="Version identifier of the prediction API model"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get(
    "/",
    tags=["General"],
    summary="API welcome message"
)
def home():
    return {
        "message": "Employee Prediction API is running",
        "documentation": "/docs",
        "health_check": "/health"
    }


# --------------------------------------------------
# 7. Health endpoint
# --------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Check API health"
)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True
    )


# --------------------------------------------------
# 8. Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Request validation failed"
        },
        500: {
            "model": ErrorResponse,
            "description": "Prediction could not be completed"
        }
    },
    tags=["Machine Learning"],
    summary="Predict employee target class",
    description=(
        "Accepts five employee features and returns the "
        "predicted class and estimated class probabilities."
    )
)
def predict_employee(employee: EmployeeData):

    try:
        employee_df = pd.DataFrame(
            [employee.model_dump()],
            columns=FEATURES
        )

        prediction = model.predict(employee_df)[0]

        probabilities = model.predict_proba(employee_df)[0]

        class_probabilities = {
            str(class_label): float(probability)
            for class_label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        logger.info("Prediction completed successfully")

        return PredictionResponse(
            predicted_class=int(prediction),
            class_probabilities=class_probabilities,
            model_version="1.1.0"
        )

    except Exception:
        # Log technical details for debugging, but don't
        # expose internal exceptions to API clients.
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed."
        )