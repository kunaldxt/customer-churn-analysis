import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List
from io import StringIO

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

# Viz
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Card-like containers */
    .css-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Section headers */
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #a78bfa;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h3 {
        color: #34495e;
        margin-top: 20px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f5f3ff 0%, #ecfdf5 100%);
        border-left: 4px solid #a78bfa;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    
    /* Divider */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #a78bfa, transparent);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
DATA_DEFAULT_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"
ID_COL = "customerID"

# Enhanced color palette - Modern Purple & Mint
CHURN_COLOR_MAP = {"No": "#a78bfa", "Yes": "#6ee7b7"}
CATEGORY_COLORS = ["#a78bfa", "#8b5cf6", "#6ee7b7", "#34d399", "#60a5fa", "#f472b6", "#fb923c", "#fbbf24"]

CATEGORICAL_COLS_DEFAULT = [
    "gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
]
NUMERIC_COLS_DEFAULT = ["tenure","MonthlyCharges","TotalCharges"]

# Prettier display names
DISPLAY_NAMES = {
    "Contract": "Contract Type",
    "InternetService": "Internet Service",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "tenure": "Tenure (months)",
    "TotalCharges": "Total Charges ($)",
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing"
}

def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace({" ": np.nan}), errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    service_no_map_cols = [
        "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]
    for c in service_no_map_cols:
        if c in df.columns:
            df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    subset_cols = [TARGET_COL] if TARGET_COL in df.columns else []
    for c in ["tenure","MonthlyCharges","TotalCharges"]:
        if c in df.columns:
            subset_cols.append(c)
    if subset_cols:
        df = df.dropna(subset=subset_cols)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_DEFAULT_PATH, uploaded: pd.DataFrame | None = None) -> pd.DataFrame:
    if uploaded is not None:
        df = uploaded.copy()
    else:
        df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace({" ": np.nan}), errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET_COL in df.columns:
        df = df[~df[TARGET_COL].isna()]
    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("---")
        st.header("🔍 Filters")
        fdf = df.copy()
        
        # Track active filters
        active_filters = []
        
        if "Contract" in fdf.columns:
            contract = st.multiselect("📄 Contract Type", sorted(fdf["Contract"].dropna().unique().tolist()))
            if contract:
                fdf = fdf[fdf["Contract"].isin(contract)]
                active_filters.append(f"Contract: {', '.join(contract)}")
        
        if "InternetService" in fdf.columns:
            internet = st.multiselect("🌐 Internet Service", sorted(fdf["InternetService"].dropna().unique().tolist()))
            if internet:
                fdf = fdf[fdf["InternetService"].isin(internet)]
                active_filters.append(f"Internet: {', '.join(internet)}")
        
        if "tenure" in fdf.columns:
            t_min, t_max = int(fdf["tenure"].min()), int(fdf["tenure"].max())
            tenure = st.slider("📅 Tenure (months)", t_min, t_max, (t_min, t_max))
            if tenure != (t_min, t_max):
                fdf = fdf[(fdf["tenure"] >= tenure[0]) & (fdf["tenure"] <= tenure[1])]
                active_filters.append(f"Tenure: {tenure[0]}-{tenure[1]} mo")
        
        if "MonthlyCharges" in fdf.columns:
            m_min, m_max = float(fdf["MonthlyCharges"].min()), float(fdf["MonthlyCharges"].max())
            monthly = st.slider("💰 Monthly Charges", m_min, m_max, (m_min, m_max))
            if monthly != (m_min, m_max):
                fdf = fdf[(fdf["MonthlyCharges"] >= monthly[0]) & (fdf["MonthlyCharges"] <= monthly[1])]
                active_filters.append(f"Charges: ${monthly[0]:.0f}-${monthly[1]:.0f}")
        
        # Display active filters summary
        if active_filters:
            st.markdown("---")
            st.markdown("**Active Filters:**")
            for f in active_filters:
                st.caption(f"• {f}")
        
        return fdf


@st.cache_resource(show_spinner=False)
def build_model(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    num_features = [c for c in num_cols if c in st.session_state.df.columns]
    cat_features = [c for c in cat_cols if c in st.session_state.df.columns]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    return pipe


def train_and_eval(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Tuple[Pipeline, dict]:
    dfp = preprocess_telco(df)

    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    if TARGET_COL in dfp.columns:
        if dfp[TARGET_COL].dtype == object:
            dfp[TARGET_COL] = dfp[TARGET_COL].str.strip().str.title()
            y = dfp[TARGET_COL].map({"Yes": 1, "No": 0})
        elif pd.api.types.is_numeric_dtype(dfp[TARGET_COL]):
            y = dfp[TARGET_COL].astype(int)
        else:
            raise ValueError("Unexpected target format")
        
        X = dfp.drop(columns=[TARGET_COL])
    else:
        raise ValueError("Target column 'Churn' not found")

    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = build_model(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    prauc = average_precision_score(y_test, y_proba)
    fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
    prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)

    metrics = {
        "roc_auc": roc,
        "pr_auc": prauc,
        "roc_curve": (fpr, tpr, roc_thr),
        "pr_curve": (prec, rec, pr_thr),
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    return pipe, metrics


def plot_roc(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        mode="lines", 
        name=f"ROC (AUC={auc_score:.3f})",
        line=dict(color="#a78bfa", width=3),
        fill='tonexty',
        fillcolor='rgba(167, 139, 250, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], 
        mode="lines", 
        name="Random Baseline", 
        line=dict(dash="dash", color="#95a5a6", width=2)
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate", 
        yaxis_title="True Positive Rate", 
        title="📊 ROC Curve",
        template="plotly_white",
        plot_bgcolor='rgba(250, 248, 255, 0)',
        hovermode='x unified',
        font=dict(size=12)
    )
    return fig


def plot_pr(prec, rec, pr_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, 
        mode="lines", 
        name=f"PR (AUC={pr_auc:.3f})",
        line=dict(color="#6ee7b7", width=3),
        fill='tonexty',
        fillcolor='rgba(110, 231, 183, 0.1)'
    ))
    fig.update_layout(
        xaxis_title="Recall", 
        yaxis_title="Precision", 
        title="📈 Precision-Recall Curve",
        template="plotly_white",
        plot_bgcolor='rgba(250, 248, 255, 0)',
        hovermode='x unified',
        font=dict(size=12)
    )
    return fig


def plot_confusion(cm, labels=("No", "Yes")):
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with counts and percentages
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)",
                    showarrow=False,
                    font=dict(color="white" if cm[i,j] > cm.max()/2 else "black", size=14)
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, 
        x=[f"Predicted<br>{l}" for l in labels], 
        y=[f"Actual<br>{l}" for l in labels],
        colorscale=[[0, '#f5f3ff'], [0.5, '#a78bfa'], [1, '#8b5cf6']],
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title="🎯 Confusion Matrix",
        annotations=annotations,
        template="plotly_white",
        xaxis=dict(side="bottom"),
        font=dict(size=12)
    )
    return fig


def create_insight_box(icon: str, title: str, message: str, color: str = "#a78bfa"):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                border-left: 4px solid {color}; 
                border-radius: 8px; 
                padding: 15px; 
                margin: 15px 0;">
        <p style="margin: 0; font-size: 1.1em;">
            <strong>{icon} {title}</strong><br>
            <span style="color: #555;">{message}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Sidebar data input
# -----------------------------
with st.sidebar:
    st.markdown("# 📉 Telco Churn")
    st.markdown("---")
    st.caption("📁 Upload your CSV or use the default dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    use_default = st.toggle("Use default dataset", value=True)

# Load data
try:
    df_uploaded = pd.read_csv(uploaded_file) if uploaded_file else None
except Exception as e:
    st.sidebar.error(f"❌ Failed to read uploaded CSV: {e}")
    df_uploaded = None

st.session_state.df = load_data(uploaded=df_uploaded) if (uploaded_file is not None) else (load_data(DATA_DEFAULT_PATH) if use_default else None)

if st.session_state.get("df") is None:
    st.warning("⚠️ No dataset loaded. Upload a CSV or enable 'Use default dataset'.")
    st.stop()

df = st.session_state.get("df")
if df is None or df.empty:
    st.error("❌ Dataset not loaded. Please upload a CSV or enable default dataset.")
    st.stop()
else:
    df = df.copy()

# -----------------------------
# Header
# -----------------------------
st.markdown("# Customer Churn Analytics Dashboard")
st.markdown("### Interactive insights, cohort analysis, and predictive modeling")
st.markdown("---")

# -----------------------------
# Filters & KPIs
# -----------------------------
fdf = filter_dataframe(df)

# Key Insights Summary
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

rows_filtered = len(fdf)
rows_total = len(df)
filter_pct = (rows_filtered / rows_total * 100) if rows_total > 0 else 0

with col1:
    st.metric(
        "📋 Total Customers", 
        f"{rows_filtered:,}",
        delta=f"{filter_pct:.0f}% of dataset" if filter_pct < 100 else None
    )

with col2:
    if TARGET_COL in fdf.columns:
        churn_rate = fdf[TARGET_COL].map({"Yes": 1, "No": 0}).mean() * 100
        # Color code based on rate
        metric_color = "🔴" if churn_rate > 30 else "🟡" if churn_rate > 20 else "🟢"
        st.metric(
            f"{metric_color} Churn Rate", 
            f"{churn_rate:.1f}%",
            delta=None
        )
    else:
        st.metric("Churn Rate", "—")

with col3:
    if 'MonthlyCharges' in fdf.columns:
        avg_monthly = fdf['MonthlyCharges'].mean()
        st.metric("💰 Avg Monthly Revenue", f"${avg_monthly:.2f}")
    else:
        st.metric("Avg Monthly Charges", "—")

with col4:
    if 'tenure' in fdf.columns:
        avg_tenure = fdf['tenure'].mean()
        st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} mo")
    else:
        st.metric("Avg Tenure", "—")

st.markdown("---")

# Quick insights
if TARGET_COL in fdf.columns:
    churned = fdf[fdf[TARGET_COL] == "Yes"]
    not_churned = fdf[fdf[TARGET_COL] == "No"]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if 'tenure' in fdf.columns:
            avg_tenure_churn = churned['tenure'].mean()
            avg_tenure_stay = not_churned['tenure'].mean()
            create_insight_box(
                "💡", 
                "Tenure Insight",
                f"Churned customers stayed {avg_tenure_churn:.1f} months on average vs {avg_tenure_stay:.1f} months for retained customers.",
                "#a78bfa"
            )
    
    with col_b:
        if 'MonthlyCharges' in fdf.columns:
            avg_charges_churn = churned['MonthlyCharges'].mean()
            avg_charges_stay = not_churned['MonthlyCharges'].mean()
            diff_pct = ((avg_charges_churn - avg_charges_stay) / avg_charges_stay * 100)
            create_insight_box(
                "📊", 
                "Revenue Insight",
                f"Churned customers paid ${avg_charges_churn:.2f}/mo vs ${avg_charges_stay:.2f}/mo ({diff_pct:+.1f}%).",
                "#6ee7b7"
            )

# -----------------------------
# Tabs
# -----------------------------
TAB_OVERVIEW, TAB_COHORTS, TAB_MODEL, TAB_SCORE = st.tabs([
    "📈 Overview", 
    "🔍 Cohort Analysis", 
    "🤖 Predictive Model", 
    "✅ Score & Export"
])

with TAB_OVERVIEW:
    st.markdown("## 📊 Distribution Analysis")
    
    with st.expander("ℹ️ About this section"):
        st.write("""
        Explore the distribution of key features in your dataset. 
        Use the filters in the sidebar to focus on specific customer segments.
        Charts show both overall distribution and breakdown by churn status.
        """)
    
    cols_plot = st.multiselect(
        "Select features to visualize",
        options=[c for c in fdf.columns if c not in [ID_COL]],
        default=[c for c in ["Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "tenure"] if c in fdf.columns]
    )
    
    for col in cols_plot:
        display_name = DISPLAY_NAMES.get(col, col)
        
        if pd.api.types.is_numeric_dtype(fdf[col]):
            fig = px.histogram(
                fdf, x=col, 
                color=TARGET_COL if TARGET_COL in fdf.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf.columns else None,
                nbins=50, 
                marginal="violin",
                opacity=0.7
            )
        else:
            fig = px.histogram(
                fdf, x=col, 
                color=TARGET_COL if TARGET_COL in fdf.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf.columns else None,
                opacity=0.8
            )
        
        fig.update_layout(
            title=f"Distribution: {display_name}",
            xaxis_title=display_name,
            yaxis_title="Count",
            template="plotly_white",
            plot_bgcolor='rgba(250, 248, 255, 0)',
            font=dict(size=12),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

with TAB_COHORTS:
    st.markdown("## 🔍 Cohort & Segment Analysis")
    
    with st.expander("ℹ️ About cohort analysis"):
        st.write("""
        Analyze churn patterns across different customer segments.
        Identify which combinations of features have the highest churn risk.
        """)
    
    st.markdown("### Churn Rate by Contract & Internet Service")
    if all(c in fdf.columns for c in ["Contract", "InternetService", TARGET_COL]):
        temp = (
            fdf.groupby(["Contract", "InternetService"], dropna=False)[TARGET_COL]
            .apply(lambda s: (s.map({"Yes":1, "No":0}).mean() * 100))
            .reset_index(name="Churn %")
        )
        
        # Add average reference line
        avg_churn = temp["Churn %"].mean()
        
        fig = px.bar(
            temp, x="Contract", y="Churn %", 
            color="InternetService", 
            barmode="group", 
            text=temp["Churn %"].round(1),
            color_discrete_sequence=CATEGORY_COLORS
        )
        
        # Add average line
        fig.add_hline(
            y=avg_churn, 
            line_dash="dash", 
            line_color="#95a5a6",
            annotation_text=f"Average: {avg_churn:.1f}%",
            annotation_position="right"
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            yaxis_title="Churn Rate (%)",
            template="plotly_white",
            plot_bgcolor='rgba(51, 48, 112, 0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Find highest risk segment
        max_churn_row = temp.loc[temp["Churn %"].idxmax()]
        create_insight_box(
            "⚠️",
            "Highest Risk Segment",
            f"{max_churn_row['Contract']} contract with {max_churn_row['InternetService']} internet has {max_churn_row['Churn %']:.1f}% churn rate.",
            "#fb923c"
        )
    
    st.markdown("---")
    st.markdown("### 💰 Monthly Charges vs Tenure Relationship")
    
    if all(c in fdf.columns for c in ["MonthlyCharges","tenure"]):
        fdf_scatter = fdf.dropna(subset=["MonthlyCharges", "tenure"]).copy()
        
        if len(fdf_scatter) >= 3:
            fig = px.scatter(
                fdf_scatter,
                x="tenure",
                y="MonthlyCharges",
                color=TARGET_COL if TARGET_COL in fdf_scatter.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf_scatter.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf_scatter.columns else None,
                hover_data=[ID_COL] if ID_COL in fdf_scatter.columns else None,
                trendline="lowess",
                marginal_x="box",
                marginal_y="violin",
                opacity=0.5
            )
        else:
            fig = px.scatter(
                fdf_scatter,
                x="tenure",
                y="MonthlyCharges",
                color=TARGET_COL if TARGET_COL in fdf_scatter.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf_scatter.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf_scatter.columns else None,
                hover_data=[ID_COL] if ID_COL in fdf_scatter.columns else None,
                opacity=0.5
            )
        
        fig.update_layout(
            xaxis_title="Tenure (months)", 
            yaxis_title="Monthly Charges ($)",
            height=550,
            template="plotly_white",
            plot_bgcolor='rgba(51, 48, 112, 0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

with TAB_MODEL:
    st.markdown("## 🤖 Predictive Modeling")
    
    with st.expander("ℹ️ What is ROC-AUC?"):
        st.write("""
        **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** measures the model's ability 
        to distinguish between churned and non-churned customers. Score ranges from 0.5 (random) to 1.0 (perfect).
        - **0.9-1.0**: Excellent
        - **0.8-0.9**: Good
        - **0.7-0.8**: Fair
        - **Below 0.7**: Poor
        """)
    
    with st.expander("ℹ️ What is PR-AUC?"):
        st.write("""
        **Precision-Recall AUC** is especially useful for imbalanced datasets (like churn).
        It focuses on the positive class (churned customers) performance. Higher is better.
        """)
    
    st.caption("⚡ Quick model training using Logistic Regression. For production, train offline and load a persisted model.")
    
    st.markdown("### Feature Selection")
    col_a, col_b = st.columns(2)
    
    with col_a:
        cat_cols = st.multiselect(
            "📋 Categorical Features", 
            options=[c for c in CATEGORICAL_COLS_DEFAULT if c in df.columns], 
            default=[c for c in CATEGORICAL_COLS_DEFAULT if c in df.columns]
        )
    
    with col_b:
        num_cols = st.multiselect(
            "🔢 Numeric Features", 
            options=[c for c in NUMERIC_COLS_DEFAULT if c in df.columns], 
            default=[c for c in NUMERIC_COLS_DEFAULT if c in df.columns]
        )

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model... ⏳"):
            pipe, metrics = train_and_eval(df, cat_cols, num_cols)
            st.session_state.model = pipe
            st.session_state.metrics = metrics
        st.success("✅ Model trained successfully!")

    if "metrics" in st.session_state:
        m = st.session_state.metrics
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            roc_color = "🟢" if m['roc_auc'] >= 0.8 else "🟡" if m['roc_auc'] >= 0.7 else "🔴"
            st.metric(f"{roc_color} ROC-AUC Score", f"{m['roc_auc']:.3f}")
        
        with perf_col2:
            pr_color = "🟢" if m['pr_auc'] >= 0.7 else "🟡" if m['pr_auc'] >= 0.5 else "🔴"
            st.metric(f"{pr_color} PR-AUC Score", f"{m['pr_auc']:.3f}")
        
        with perf_col3:
            test_size = len(m['y_test'])
            st.metric("🧪 Test Set Size", f"{test_size:,}")
        
        # Performance interpretation
        if m['roc_auc'] >= 0.8:
            create_insight_box(
                "🎉",
                "Excellent Model Performance",
                f"This model achieves {m['roc_auc']:.1%} ROC-AUC, indicating strong predictive power for identifying churn risk.",
                "#6ee7b7"
            )
        elif m['roc_auc'] >= 0.7:
            create_insight_box(
                "👍",
                "Good Model Performance",
                f"Your model achieves {m['roc_auc']:.1%} ROC-AUC. Consider feature engineering or trying other algorithms for improvement.",
                "#fbbf24"
            )
        else:
            create_insight_box(
                "⚠️",
                "Model Needs Improvement",
                f"ROC-AUC of {m['roc_auc']:.1%} suggests limited predictive power. Try adding more features or collecting more data.",
                "#fb923c"
            )
        
        st.markdown("### 📊 Performance Curves")
        curve_col1, curve_col2 = st.columns(2)
        
        with curve_col1:
            fpr, tpr, _ = m["roc_curve"]
            st.plotly_chart(plot_roc(fpr, tpr, m['roc_auc']), use_container_width=True)
        
        with curve_col2:
            prec, rec, _ = m["pr_curve"]
            st.plotly_chart(plot_pr(prec, rec, m['pr_auc']), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Confusion Matrix Analysis")
        
        with st.expander("ℹ️ How to read the confusion matrix"):
            st.write("""
            - **False Negatives (top-left)**: Missed churners (high cost!)
            - **True Positives (top-right)**:  Correctly identified churners
            - **True Negatives (bottom-left)**: Correctly predicted non-churners
            - **False Positives (bottom-right)**: Incorrectly predicted as churners
            """)
        
        thr = st.slider(
            "🎚️ Prediction Threshold", 
            0.0, 1.0, 0.30, 0.01,
            help="Lower threshold = catch more churners but more false alarms. Higher threshold = fewer false alarms but miss more churners."
        )
        
        y_pred = (m["y_proba"] >= thr).astype(int)
        cm = confusion_matrix(m["y_test"], y_pred)
        
        # Calculate metrics at this threshold
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Display threshold metrics
        thresh_col1, thresh_col2, thresh_col3, thresh_col4 = st.columns(4)
        
        with thresh_col1:
            st.metric("🎯 Accuracy", f"{accuracy:.1%}")
        with thresh_col2:
            st.metric("🔍 Precision", f"{precision:.1%}")
        with thresh_col3:
            st.metric("📊 Recall", f"{recall:.1%}")
        with thresh_col4:
            st.metric("⚖️ F1 Score", f"{f1:.1%}")
        
        st.plotly_chart(plot_confusion(cm), use_container_width=True)
        
        # Business impact
        total_predicted_churn = fp + tp
        cost_per_intervention = 100  # Example
        value_saved_per_retained = 500  # Example
        
        estimated_cost = total_predicted_churn * cost_per_intervention
        estimated_value = tp * value_saved_per_retained
        net_value = estimated_value - estimated_cost
        
        st.markdown("### 💼 Business Impact Estimation")
        st.caption("Example calculation based on assumed intervention costs and customer lifetime value")
        
        impact_col1, impact_col2, impact_col3 = st.columns(3)
        
        with impact_col1:
            st.metric("💰 Intervention Cost", f"${estimated_cost:,.0f}", 
                     delta=f"{total_predicted_churn} customers @ ${cost_per_intervention}")
        with impact_col2:
            st.metric("💎 Value Saved", f"${estimated_value:,.0f}",
                     delta=f"{tp} retained @ ${value_saved_per_retained}")
        with impact_col3:
            net_color = "normal" if net_value > 0 else "inverse"
            st.metric("📈 Net Value", f"${net_value:,.0f}",
                     delta="Positive ROI" if net_value > 0 else "Negative ROI",
                     delta_color=net_color)

with TAB_SCORE:
    st.markdown("## ✅ Score Dataset & Export Results")
    
    with st.expander("ℹ️ About scoring"):
        st.write("""
        Apply the trained model to the current filtered dataset to generate churn predictions.
        Export results as CSV for further analysis or integration into business processes.
        """)
    
    if "model" not in st.session_state:
        create_insight_box(
            "ℹ️",
            "Model Required",
            "Please train a model in the 'Predictive Model' tab before scoring customers.",
            "#a78bfa"
        )
    else:
        mdl: Pipeline = st.session_state.model
        
        st.markdown("### 🎚️ Configure Scoring Threshold")
        thr2 = st.slider(
            "Prediction threshold for flagging high-risk customers", 
            0.0, 1.0, 0.30, 0.01, 
            key="thr2",
            help="Customers with predicted probability above this threshold will be flagged as high churn risk"
        )
        
        # Score the dataset
        X_score = fdf.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
        proba = mdl.predict_proba(X_score)[:, 1]
        pred = (proba >= thr2).astype(int)
        
        # Calculate scoring statistics
        high_risk_count = pred.sum()
        high_risk_pct = (high_risk_count / len(pred) * 100) if len(pred) > 0 else 0
        
        st.markdown("### 📊 Scoring Summary")
        score_col1, score_col2, score_col3 = st.columns(3)
        
        with score_col1:
            st.metric("👥 Total Customers Scored", f"{len(pred):,}")
        with score_col2:
            risk_color = "🔴" if high_risk_pct > 30 else "🟡" if high_risk_pct > 15 else "🟢"
            st.metric(f"{risk_color} High Risk Customers", f"{high_risk_count:,}")
        with score_col3:
            st.metric("📊 High Risk %", f"{high_risk_pct:.1f}%")
        
        # Build scored output
        base_cols = []
        if ID_COL in fdf.columns:
            base_cols.append(ID_COL)
        if TARGET_COL in fdf.columns:
            base_cols.append(TARGET_COL)
        
        scored = fdf[base_cols].copy() if base_cols else pd.DataFrame(index=fdf.index)
        scored["churn_probability"] = proba
        scored["churn_prediction"] = pred
        
        # Dynamic risk categories based on threshold
        # Low Risk: < 50% of threshold
        # Medium Risk: 50% of threshold to threshold
        # High Risk: >= threshold
        low_cutoff = thr2 * 0.5
        scored["risk_category"] = pd.cut(
            proba, 
            bins=[0, low_cutoff, thr2, 1.0], 
            labels=["Low Risk", "Medium Risk", "High Risk"],
            include_lowest=True
        )
        
        # Add key features for context
        for col in ["Contract", "tenure", "MonthlyCharges"]:
            if col in fdf.columns:
                scored[col] = fdf[col].values
        
        st.markdown("---")
        st.markdown("### 📋 Scored Customer Data (Top 50)")
        
        # Show distribution by risk category
        if "risk_category" in scored.columns:
            risk_dist = scored["risk_category"].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(
                x=risk_dist.index.astype(str),
                y=risk_dist.values,
                text=risk_dist.values,
                textposition='auto',
                marker_color=['#a78bfa', '#fbbf24', '#fb923c']
            )])
            
            fig.update_layout(
                title="Risk Category Distribution",
                xaxis_title="Risk Level",
                yaxis_title="Number of Customers",
                template="plotly_white",
                plot_bgcolor='rgba(250, 248, 255, 0)',
                showlegend=False,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display styled dataframe
        st.dataframe(
            scored.head(50).style.background_gradient(
                subset=['churn_probability'], 
                cmap='RdYlGn_r',
                vmin=0,
                vmax=1
            ),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        st.markdown("### 💾 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Full export
            csv_full = scored.to_csv(index=False)
            st.download_button(
                "📥 Download Full Scored Dataset",
                data=csv_full,
                file_name="telco_churn_scored_full.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
        
        with export_col2:
            # High risk only
            high_risk_df = scored[scored["churn_prediction"] == 1]
            csv_high_risk = high_risk_df.to_csv(index=False)
            st.download_button(
                "🚨 Download High Risk Only",
                data=csv_high_risk,
                file_name="telco_churn_high_risk.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        create_insight_box(
            "💡",
            "Next Steps",
            "Use the scored dataset to: (1) Prioritize retention campaigns, (2) Identify at-risk segments, (3) Calculate intervention ROI, (4) Monitor trends over time.",
            "#6ee7b7"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p style="margin: 5px 0;">💡 <strong>Pro Tip:</strong> Save trained models with joblib and load them for faster scoring in production</p>
    <p style="margin: 5px 0;">📚 Built with Streamlit • Scikit-learn • Plotly</p>
    <p style="margin: 5px 0; font-size: 0.9em;">Telco Customer Churn Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)