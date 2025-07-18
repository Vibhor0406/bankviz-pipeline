# 🏦 BankViz Pipeline

**BankViz Pipeline** is an end-to-end machine learning and visualization project focused on predicting customer deposit behavior in banking datasets. This pipeline performs data preprocessing, model training, evaluation, and generates interactive visualizations to provide actionable business insights.

---

## 📌 Features

- Automated ingestion of bank marketing data
- Categorical encoding and feature scaling
- Logistic Regression modeling with performance evaluation
- Interactive **ROC Curve** using Plotly
- Model explainability with **SHAP** value visualization
- Seamless UI integration via the `preswald` module

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `pandas`, `scikit-learn` – for data manipulation and ML
- `plotly` – for interactive visualizations
- `shap` – for model interpretability
- `preswald` – for UI/data binding (custom or external module)

---

## 📁 Project Structure


---

## 🔄 Workflow Overview

1. **Data Load**: Imports `bank.csv` via `preswald.get_df()`
2. **Preprocessing**:
   - Label Encoding for categorical variables
   - Feature scaling with `StandardScaler`
3. **Model Training**: Logistic Regression
4. **Evaluation**: ROC Curve + AUC calculation
5. **Visualization**:
   - ROC Curve via `plotly.graph_objects`
   - Feature importance via `shap.Explainer`

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bankviz-pipeline.git
cd bankviz-pipeline

### 2. Install dependencies

```bash
pip install -r requirements.txt

### 3. Run the pipeline

```bash
python hello.py
