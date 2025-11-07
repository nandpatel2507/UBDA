import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

def find_label_column(df):
    for c in df.columns:
        if c.lower().replace(" ", "") in ["personalloan","personal_loan"]:
            return c
    return None

def load_and_prepare_data(path_or_df):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()
    # Drop ID and Zip code if present
    cols = [c for c in df.columns]
    if "ID" in cols:
        df = df.drop(columns=["ID"])
    if "Zip code" in cols:
        df = df.drop(columns=["Zip code"])
    label = find_label_column(df)
    if label is None:
        raise ValueError("Label column 'Personal Loan' not found.")
    y = df[label].astype(int)
    X = df.drop(columns=[label])
    return X, y, df, label

def train_all_models(df, label_col=None, random_state=42, n_estimators=100, cv=5):
    if label_col is None:
        label_col = find_label_column(df)
    X, y, _, _ = load_and_prepare_data(df if isinstance(df, str) else df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
        "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    }
    metrics = []
    probs_test = {}
    confusion_matrices = {}
    feature_importances = {}
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state), scoring='accuracy', n_jobs=1)
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except Exception:
            cv_mean, cv_std = (np.nan, np.nan)
        model.fit(X_train, y_train)
        fitted = model
        y_train_pred = fitted.predict(X_train)
        y_test_pred = fitted.predict(X_test)
        y_train_prob = fitted.predict_proba(X_train)[:,1] if hasattr(fitted, "predict_proba") else None
        y_test_prob = fitted.predict_proba(X_test)[:,1] if hasattr(fitted, "predict_proba") else None
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_prob) if y_test_prob is not None else np.nan
        metrics.append({
            "Algorithm": name,
            "Training Accuracy": train_acc,
            "Testing Accuracy": test_acc,
            "Precision (test)": test_prec,
            "Recall (test)": test_rec,
            "F1-score (test)": test_f1,
            "AUC (test)": test_auc,
            "CV-{} Accuracy Mean".format(cv): cv_mean,
            "CV-{} Accuracy Std".format(cv): cv_std
        })
        probs_test[name] = y_test_prob
        confusion_matrices[name] = {"train": confusion_matrix(y_train, y_train_pred), "test": confusion_matrix(y_test, y_test_pred)}
        if hasattr(fitted, "feature_importances_"):
            feature_importances[name] = {"features": list(X.columns), "importances": fitted.feature_importances_}
        else:
            feature_importances[name] = {"features": list(X.columns), "importances": np.zeros(X.shape[1])}
    metrics_df = pd.DataFrame(metrics).set_index("Algorithm").round(4)
    return {
        "metrics_df": metrics_df,
        "probs_test": probs_test,
        "y_test": y_test,
        "confusion_matrices": confusion_matrices,
        "feature_importances": feature_importances
    }

def plot_roc_plotly(probs_dict, y_test):
    fig = go.Figure()
    colors = {"Decision Tree":"green", "Random Forest":"blue", "Gradient Boosted Tree":"red"}
    for name, probs in probs_dict.items():
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc:.3f})", line=dict(color=colors.get(name,None))))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Chance', line=dict(dash='dash', color='gray')))
    fig.update_layout(title="ROC Curves (Test set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def plot_confusion_plotly(cm, title="Confusion Matrix"):
    labels = ["Not Loan (0)","Loan (1)"]
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, hoverongaps=False, colorscale="Blues", showscale=True))
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(dict(x=labels[j], y=labels[i], text=str(int(cm[i,j])), showarrow=False, font=dict(color="white" if cm[i,j]>cm.max()/2 else "black")))
    fig.update_layout(title=title, annotations=annotations)
    return fig

def plot_feature_importance_plotly(features, importances, title="Feature Importances"):
    imp = np.array(importances)
    idx = np.argsort(imp)[::-1]
    fig = go.Figure(data=[go.Bar(x=[features[i] for i in idx], y=imp[idx])])
    fig.update_layout(title=title, xaxis_tickangle=-45)
    return fig

def plot_income_acceptance(df, label_col="Personal Loan", bins=8):
    dfc = df.copy()
    for c in ["ID","Zip code"]:
        if c in dfc.columns:
            dfc = dfc.drop(columns=[c])
    dfc = dfc.dropna(subset=["Income", label_col])
    dfc["income_bin"] = pd.qcut(dfc["Income"], q=bins, duplicates="drop")
    grouped = dfc.groupby("income_bin").agg(total=("Income","count"), acceptance=(label_col,"mean"), avg_income=("Income","mean"))
    grouped = grouped.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped["income_bin"].astype(str), y=grouped["total"], name="Count"))
    fig.add_trace(go.Line(x=grouped["income_bin"].astype(str), y=grouped["acceptance"], name="Acceptance Rate", yaxis="y2"))
    fig.update_layout(title="Income Bins: Count and Acceptance Rate", yaxis=dict(title="Count"), yaxis2=dict(title="Acceptance Rate", overlaying="y", side="right", tickformat=".0%"))
    return fig

def plot_scatter_income_ccavg(df, label_col="Personal Loan"):
    dfc = df.copy().dropna(subset=["Income","CCAvg", label_col])
    fig = px.scatter(dfc, x="Income", y="CCAvg", color=label_col, size="Age" if "Age" in dfc.columns else None,
                     hover_data=[c for c in dfc.columns if c not in ["Income","CCAvg"]], title="Income vs CCAvg (colored by Personal Loan)")
    return fig

def plot_edu_family_stacked(df, label_col="Personal Loan"):
    dfc = df.copy().dropna(subset=["Education","Family", label_col])
    pivot = dfc.groupby(["Education","Family"])[label_col].mean().reset_index()
    pivot["Education"] = pivot["Education"].astype(str)
    fig = px.bar(pivot, x="Education", y=label_col, color="Family", barmode="group", labels={label_col:"Acceptance Rate"}, title="Acceptance Rate by Education and Family Size")
    return fig

def plot_pca_scatter(df, label_col="Personal Loan"):
    X, y, _, _ = load_and_prepare_data(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    comp = pca.fit_transform(Xs)
    d = pd.DataFrame(comp, columns=["PC1","PC2"])
    d[label_col] = y.values
    fig = px.scatter(d, x="PC1", y="PC2", color=label_col, title="PCA 2D Projection of Customers (colored by Personal Loan)")
    return fig

def plot_lift_chart_plotly(df, label_col="Personal Loan"):
    X, y, _, _ = load_and_prepare_data(df)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)[:,1]
    dfc = pd.DataFrame({"y": y, "score": probs})
    dfc = dfc.sort_values("score", ascending=False).reset_index(drop=True)
    dfc["cum_response"] = dfc["y"].cumsum()
    total_positives = dfc["y"].sum()
    dfc["cum_pct_customers"] = (dfc.index + 1) / len(dfc)
    dfc["cum_pct_positives"] = dfc["cum_response"] / total_positives if total_positives>0 else 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfc["cum_pct_customers"], y=dfc["cum_pct_positives"], mode="lines", name="Cumulative Gain"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="Cumulative Gain Chart", xaxis_title="Proportion of Customers Contacted", yaxis_title="Proportion of Positive Responses (Gain)")
    return fig

def preprocess_new_data(train_df, new_df):
    X_train, y_train, _, _ = load_and_prepare_data(train_df)
    new = new_df.copy()
    for c in ["ID","Zip code"]:
        if c in new.columns:
            new = new.drop(columns=[c])
    common = [c for c in X_train.columns if c in new.columns]
    X_new = new[common].copy()
    for c in X_train.columns:
        if c not in X_new.columns:
            X_new[c] = 0
    X_new = X_new[X_train.columns]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, {"X_new": X_new}
