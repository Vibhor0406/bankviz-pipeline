from preswald import connect, get_df, text, plotly
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
import shap

connect()
df = get_df("bank.csv")

# Label encode and scale
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object'):
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("deposit", axis=1)
y = df_encoded["deposit"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}"))
fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")

text("# Bank ML App")
text("## Logistic Regression ROC Curve")
plotly(fig)

# SHAP Feature Importance
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=10)  # This runs in notebook; not supported in Preswald