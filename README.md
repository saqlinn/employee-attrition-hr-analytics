# Employee Attrition Prediction & HR Analytics Platform

An end-to-end Machine Learning project that analyzes employee data, performs classification and clustering, and provides an interactive HR analytics dashboard.

🔗 **GitHub Repository:** https://github.com/saqlinn/employee-attrition-hr-analytics

## 🎯 Project Overview

The project uses machine learning to classify employee records, explore workforce patterns, and group similar records for analytical purposes.

### ✨ Key Features
- Exploratory Data Analysis (EDA) and data visualization.
- Random Forest classification.
- K-Means employee segmentation.
- FastAPI prediction API with Swagger documentation.
- Interactive Streamlit dashboard.
- API testing using Pytest.

## 🛠️ Tech Stack

**Python | Pandas | NumPy | Scikit-learn | Random Forest | K-Means | FastAPI | Streamlit | Plotly | Joblib | Pytest**

## 📊 Dataset

The dataset contains **1,000 employee records and 8 columns**, including work experience, organizations worked at, feedback, promotion-related information, and an `Attrition` target variable.

## 🏗️ Project Structure

```text
Employee-Attrition-Project/
├── api/          # FastAPI backend
├── dashboard/    # Streamlit dashboard
├── data/         # Employee datasets
├── models/       # Saved ML model and metadata
├── notebooks/    # Analysis and experiments
├── src/          # Prediction utilities
├── tests/        # API tests
├── requirements.txt
└── README.md
```

## 🚀 Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/saqlinn/employee-attrition-hr-analytics.git
cd employee-attrition-hr-analytics
```

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**4. Start the FastAPI backend**
```bash
python -m uvicorn api.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

**5. Start the Streamlit dashboard** in a second terminal:
```bash
python -m streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

**6. Run API tests**
```bash
python -m pytest tests/test_api.py -v
```

## 🔮 Future Improvements

- Improve model evaluation and feature selection.
- Add model explainability.
- Persist clustering models for consistent predictions.
- Deploy the application online.
- Add database integration and monitoring.

## 👨‍💻 Author

**Saqlin Mustaq**  
B.Tech — Artificial Intelligence and Data Science

- GitHub: https://github.com/saqlinn
- LinkedIn: https://www.linkedin.com/in/saqlin-mustaq-94021129a/

---
