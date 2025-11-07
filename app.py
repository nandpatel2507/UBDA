import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_and_prepare_data, train_all_models, compute_model_metrics,
    plot_roc_plotly, plot_confusion_plotly, plot_feature_importance_plotly,
    plot_income_acceptance, plot_scatter_income_ccavg, plot_edu_family_stacked,
    plot_pca_scatter, plot_lift_chart_plotly, preprocess_new_data
)
st.set_page_config(layout="wide", page_title="Universal Bank - Loan Prediction Dashboard")

st.title("Universal Bank — Personal Loan Prediction & Marketing Insights")
st.markdown("Dashboard to explore customer data, train tree models, and predict personal loan interest.")

@st.cache_data
def load_default():
    try:
        df = pd.read_csv("UniversalBank.csv")
        return df
    except Exception as e:
        return None

df_default = load_default()

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload Universal Bank CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("Uploaded dataset loaded")
    else:
        if df_default is not None:
            df = df_default.copy()
            st.info("Using included UniversalBank.csv")
        else:
            st.warning("No dataset found. Please upload a CSV on the Upload tab.")
            df = None

tabs = st.tabs(["Dashboard", "Model Trainer", "Predict & Upload", "Readme / Guide"])

def find_label_column(df):
    for c in df.columns:
        if c.lower().replace(" ", "") in ["personalloan","personal_loan"]:
            return c
    return None

with tabs[0]:
    st.header("Marketing Insights & Charts")
    if df is None:
        st.warning("Please upload dataset in the sidebar or on the Predict & Upload tab.")
    else:
        label = find_label_column(df)
        if label is None:
            st.error("Couldn't find the 'Personal Loan' label column. Ensure column is named 'Personal Loan' (case-insensitive).")
        else:
            st.subheader("Dataset snapshot")
            st.dataframe(df.head())

            st.markdown("### Charts (actionable insights for Marketing Head)")
            c1, c2 = st.columns([1,1])
            with c1:
                st.markdown("**1. Income bins & Acceptance rate**")
                fig1 = plot_income_acceptance(df, label_col=label, bins=8)
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                st.markdown("**2. Income vs CCAvg scatter (targeted offers)**")
                fig2 = plot_scatter_income_ccavg(df, label_col=label)
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns([1,1])
            with c3:
                st.markdown("**3. Education x Family stacked acceptance (segmentation)**")
                fig3 = plot_edu_family_stacked(df, label_col=label)
                st.plotly_chart(fig3, use_container_width=True)
            with c4:
                st.markdown("**4. PCA 2D customer clusters (visual segmentation)**")
                fig4 = plot_pca_scatter(df, label_col=label)
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("**5. Lift / Gain chart (prioritization for campaigns)**")
            fig5 = plot_lift_chart_plotly(df, label_col=label)
            st.plotly_chart(fig5, use_container_width=True)

with tabs[1]:
    st.header("Train & Evaluate Models (Decision Tree, Random Forest, Gradient Boosted)")
    if df is None:
        st.warning("Please upload dataset in the sidebar or on the Predict & Upload tab.")
    else:
        label = find_label_column(df)
        if label is None:
            st.error("Couldn't find the 'Personal Loan' label column. Ensure column is named 'Personal Loan'.")
        else:
            st.write("Models will be trained on 70% train / 30% test split with Stratified CV (5-fold).")
            if st.button("Train all models and compute metrics"):
                with st.spinner("Training models... this may take a short while"):
                    results = train_all_models(df, label_col=label, random_state=42, n_estimators=100, cv=5)
                metrics_df = results["metrics_df"]
                st.subheader("Performance Table (rounded)")
                st.dataframe(metrics_df.style.format("{:.4f}"))

                st.subheader("ROC Curves (Test set)")
                figroc = plot_roc_plotly(results["probs_test"], results["y_test"])
                st.plotly_chart(figroc, use_container_width=True)

                st.subheader("Confusion Matrices (Train & Test)")
                for name, cms in results["confusion_matrices"].items():
                    st.markdown(f"**{name} - Training**")
                    st.plotly_chart(plot_confusion_plotly(cms["train"], title=f"{name} - Train Confusion"), use_container_width=True)
                    st.markdown(f"**{name} - Testing**")
                    st.plotly_chart(plot_confusion_plotly(cms["test"], title=f"{name} - Test Confusion"), use_container_width=True)

                st.subheader("Feature Importances")
                for name, imp in results["feature_importances"].items():
                    st.markdown(f"**{name}**")
                    st.plotly_chart(plot_feature_importance_plotly(imp["features"], imp["importances"], title=f"{name} Feature Importances"), use_container_width=True)

with tabs[2]:
    st.header("Upload new dataset & predict Personal Loan label")
    st.markdown("Upload a CSV with the same columns (ID optional). The app will drop ID and Zip code if present and add a column 'Predicted_PersonalLoan' with 0/1.")
    upload2 = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict")
    if upload2 is not None:
        newdf = pd.read_csv(upload2)
        st.write("Preview of uploaded data:")
        st.dataframe(newdf.head())

        if st.button("Predict using Random Forest (trained on full dataset)"):
            with st.spinner("Training Random Forest on full dataset and predicting..."):
                model, processed = preprocess_new_data(df, newdf)  # train model on df and predict on newdf
                preds = model.predict(processed["X_new"])
                prob = model.predict_proba(processed["X_new"])[:,1] if hasattr(model, "predict_proba") else None
                out = newdf.copy()
                out["Predicted_PersonalLoan"] = preds.astype(int)
                if prob is not None:
                    out["Pred_Prob"] = prob
                st.success("Predictions added to table below")
                st.dataframe(out.head())
                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

with tabs[3]:
    st.header("Readme / Quick Guide")
    st.markdown(open("README.md").read() if True else "")
