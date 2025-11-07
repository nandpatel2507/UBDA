import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

st.set_page_config(page_title="Universal Bank Loan Prediction", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    return df

def train_models(df):
    X = df.drop(["Personal Loan", "ID", "ZIP Code"], axis=1, errors="ignore")
    y = df["Personal Loan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)
    }

    metrics, roc_data, conf_matrices = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics.append({
            "Algorithm": name,
            "Train Accuracy": accuracy_score(y_train, y_pred_train),
            "Test Accuracy": accuracy_score(y_test, y_pred_test),
            "Precision": precision_score(y_test, y_pred_test),
            "Recall": recall_score(y_test, y_pred_test),
            "F1-Score": f1_score(y_test, y_pred_test),
            "AUC": roc_auc_score(y_test, y_prob)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)
        conf_matrices[name] = (confusion_matrix(y_train, y_pred_train), confusion_matrix(y_test, y_pred_test))
    return models, pd.DataFrame(metrics), roc_data, conf_matrices, X_train.columns

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    st.pyplot(fig)

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["📊 Insights", "🤖 Model Training", "📂 Predict New Data"])

df = load_data()

if tab == "📊 Insights":
    st.title("📊 Customer Insights Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income by Loan Status")
        fig, ax = plt.subplots(); sns.boxplot(x="Personal Loan", y="Income", data=df, palette="Set2", ax=ax); st.pyplot(fig)
    with col2:
        st.subheader("Credit Card Spend vs Education")
        fig, ax = plt.subplots(); sns.barplot(x="Education", y="CCAvg", hue="Personal Loan", data=df, ax=ax, palette="viridis"); st.pyplot(fig)

    st.subheader("Family Size and Loan Acceptance")
    fig, ax = plt.subplots(); sns.countplot(x="Family", hue="Personal Loan", data=df, palette="coolwarm", ax=ax); st.pyplot(fig)

    st.subheader("Mortgage vs Income")
    fig, ax = plt.subplots(); sns.scatterplot(x="Income", y="Mortgage", hue="Personal Loan", data=df, palette="plasma", ax=ax); st.pyplot(fig)

    st.subheader("Online Banking Usage and Loan Uptake")
    fig, ax = plt.subplots(); sns.barplot(x="Online", y="Personal Loan", data=df, estimator=np.mean, palette="magma", ax=ax); st.pyplot(fig)

elif tab == "🤖 Model Training":
    st.title("🤖 Model Training and Evaluation")
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            models, metrics_df, roc_data, conf_matrices, features = train_models(df)
        st.subheader("Model Metrics"); st.dataframe(metrics_df)

        st.subheader("ROC Curves")
        fig, ax = plt.subplots()
        for name, (fpr, tpr) in roc_data.items():
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(); st.pyplot(fig)

        st.subheader("Confusion Matrices")
        for name, (cm_train, cm_test) in conf_matrices.items():
            st.write(f"**{name} - Training**"); plot_confusion_matrix(cm_train, f"{name} Train")
            st.write(f"**{name} - Testing**"); plot_confusion_matrix(cm_test, f"{name} Test")

        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8,5))
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                sns.barplot(x=model.feature_importances_, y=features, ax=ax, label=name)
        ax.legend(); ax.set_title("Feature Importance"); st.pyplot(fig)

elif tab == "📂 Predict New Data":
    st.title("📂 Predict on New Data")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        new_df = pd.read_csv(file)
        st.dataframe(new_df.head())
        models, _, _, _, _ = train_models(df)
        choice = st.selectbox("Select Model", models.keys())
        model = models[choice]
        X_new = new_df.drop(["ID", "Personal Loan"], axis=1, errors="ignore")
        preds = model.predict(X_new)
        new_df["Predicted Personal Loan"] = preds
        st.dataframe(new_df.head())
        st.download_button("Download Predictions", new_df.to_csv(index=False).encode(), "predictions.csv", "text/csv")
