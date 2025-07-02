
---

## 🧠 Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability
Basel II requires banks to adopt internal risk measurement systems that are transparent, auditable, and explainable. Hence, we need interpretable models (e.g., Logistic Regression with Weight of Evidence) that offer traceable decision logic and align with regulatory compliance. This ensures regulators and auditors can verify how credit risk decisions are made.

### 2. Necessity of a Proxy Target
Our dataset lacks an explicit "default" label. To proceed, we construct a **proxy target** by identifying disengaged users using RFM (Recency, Frequency, Monetary) analysis. However, this introduces **business risks** — including label noise and misclassification — which may impact model accuracy and lead to unfair lending decisions if not handled carefully.

### 3. Simple vs. Complex Models: The Trade-off
- **Simple (e.g., Logistic Regression + WoE):** Easier to explain and deploy in regulated environments. Good for baseline.
- **Complex (e.g., XGBoost):** Typically offers better accuracy but lacks transparency. May be restricted in high-stakes credit environments due to low interpretability.

---

## 📊 Task 2 - Exploratory Data Analysis (EDA)

Conducted in Jupyter notebooks, the EDA revealed:

- Skewed distributions in monetary features.
- Missing values in some categorical variables.
- Several strong correlations across frequency-related features.
- Outliers in transaction amounts.
- Uneven distribution in categorical features like `ProductCategory` and `CountryCode`.

---

## 🧱 Task 3 - Feature Engineering

Implemented via `sklearn.pipeline.Pipeline` in `src/data_processing.py`. Steps included:

- Aggregate features (Total, Average, Std of transactions).
- Temporal features (Transaction hour, month, year).
- One-Hot and Label Encoding for categorical variables.
- Imputation for missing data.
- Feature scaling (Standardization).
- Optional WoE transformation using `xverse` and `woe` libraries.

---

## 🏷️ Task 4 - Proxy Target Variable Engineering

- **RFM Analysis:** Computed Recency, Frequency, and Monetary values per customer.
- **Clustering:** Applied KMeans to segment customers into 3 clusters using scaled RFM.
- **High-Risk Proxy:** Defined the least-engaged cluster as high-risk (`is_high_risk = 1`).
- **Integration:** Merged the `is_high_risk` target column into the model training dataset.

---

## 🤖 Task 5 - Model Training & Experiment Tracking

- Used **MLflow** for experiment tracking and model registry.
- Trained and compared models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost)
- Evaluated using:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Hyperparameter tuning via `hyperopt`
- Unit tests created in `tests/test_data_processing.py` using `pytest`.

---

## 🚀 Task 6 - Model Deployment & CI/CD

- Built a **REST API** using FastAPI in `src/api/main.py`
- `/predict` endpoint returns credit risk probability.
- Pydantic models used for input/output validation.
- Containerized the service with:
  - `Dockerfile`
  - `docker-compose.yml`
- CI Pipeline with GitHub Actions:
  - Code linting via `flake8`
  - Test execution via `pytest`
  - Pipeline fails on lint/test errors.

---

## 📚 References

- [Basel II Capital Accord](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring by HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Credit Risk Model Tutorial](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Credit Risk - Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer - Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)

---

## 🛠 Tech Stack

- Python, Scikit-learn, XGBoost, Pandas, NumPy
- MLflow, FastAPI, Docker, GitHub Actions
- Jupyter, Matplotlib, Seaborn
- Pytest, Flake8, Black

---


