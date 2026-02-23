import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import plotly.io as pio

st.set_page_config(page_title="AI Mastery Predictor", layout="wide")

# Page Navigation State
if "page" not in st.session_state:
    st.session_state.page = "input"

if "result_data" not in st.session_state:
    st.session_state.result_data = None

st.markdown("""
<style>

/* Main gradient background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
}

/* Glass effect container */
.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* Make text white for visibility */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

/* Improve buttons */
.stButton>button {
    border-radius: 12px;
    border: none;
    padding: 0.5rem 1rem;
    background: linear-gradient(90deg, #00C6FF, #0072FF);
    color: white;
    font-weight: 600;
}

/* Slight hover effect */
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# PAGE CONFIG
# ==========================================
st.title("🎓 LearnSight AI")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("data/digital_learning_analytics_100k.csv")

df = load_data()

model = joblib.load("models/mastery_model.pkl")
scaler = joblib.load("models/scaler.pkl")

model_features = list(scaler.feature_names_in_)

# ==========================================
# FEATURE ENGINEERING
# ==========================================
df_model = df.copy()

if "enrollment_date" in df_model.columns:
    df_model["enrollment_date"] = pd.to_datetime(df_model["enrollment_date"], errors="coerce")
    df_model["enrollment_date_year"] = df_model["enrollment_date"].dt.year
    df_model["enrollment_date_month"] = df_model["enrollment_date"].dt.month
    df_model["enrollment_date_day"] = df_model["enrollment_date"].dt.day

if "last_activity_date" in df_model.columns:
    df_model["last_activity_date"] = pd.to_datetime(df_model["last_activity_date"], errors="coerce")
    df_model["last_activity_date_year"] = df_model["last_activity_date"].dt.year
    df_model["last_activity_date_month"] = df_model["last_activity_date"].dt.month
    df_model["last_activity_date_day"] = df_model["last_activity_date"].dt.day

# ==========================================
# COLUMN TYPES
# ==========================================
numeric_cols = df_model.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_model.select_dtypes(exclude=np.number).columns.tolist()

numeric_cols = [col for col in numeric_cols if col in model_features]
categorical_cols = [col for col in categorical_cols if col in model_features]

# ==========================================
# PREPROCESS
# ==========================================
def preprocess(input_df):

    input_df = input_df.copy()
    input_df = input_df[model_features]

    for col in input_df.columns:
        if col in categorical_cols:
            if col in df_model.columns:
                mapping = {
                    val: idx
                    for idx, val in enumerate(df_model[col].dropna().unique())
                }
                input_df[col] = input_df[col].map(mapping)

            input_df[col] = input_df[col].fillna(0)

    input_df = input_df.astype(float)
    return scaler.transform(input_df)

# ==========================================
# VISUALIZATION FUNCTION
# ==========================================
def show_results(input_df):

    processed = preprocess(input_df)
    prediction = model.predict(processed)[0]

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Mastery Score", f"{prediction:.2f}")

    with col2:
        # 🎯 Dynamic color based on prediction value
        if prediction <= 40:
            bar_color = "#FF4C4C"   # Bright Coral
        elif 40 < prediction <= 70:
            bar_color = "#FF8C00"   # Bright Orange
        else:
            bar_color = "#00FF9C"   # Neon Mint

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'font': {'color': "white", 'size': 60}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 40], 'color': "#8B0000"},
                    {'range': [40, 70], 'color': "#B8860B"},
                    {'range': [70, 100], 'color': "#006400"},
                ],
            }
        ))

        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            transition_duration=800
        )

        st.plotly_chart(gauge, use_container_width=True)



    # Feature Importance
    st.markdown("### 🔝 Top 10 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_bar = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
    )

    fig_bar.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig_bar.update_layout(
        title={
            "text": "🚀 Top 10 Feature Importance",
            "x": 0.5,
            "xanchor": "center"
        },
        yaxis=dict(categoryorder="total ascending"),
        height=500,
        transition_duration=800
    )

    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    fig_bar.update_traces(
        marker_color="#00C6FF"
    )

    st.plotly_chart(fig_bar, use_container_width=True)


    # Radar Chart
    st.markdown("### 🧠 Engagement Profile")

    radar_features = [
        "forum_posts",
        "assignment_submission_rate",
        "video_completion_pct",
        "in_app_quiz_score"
    ]

    radar_data = []
    radar_labels = []

    for feat in radar_features:
        if feat in input_df.columns:
            radar_labels.append(feat)
            radar_data.append(float(input_df[feat].values[0]))

    if radar_data:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=radar_labels,
            fill='toself'
        ))

        fig_radar.update_layout(
            transition_duration=800,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    gridcolor="rgba(255,255,255,0.2)",
                    tickfont=dict(color="white")
                ),
                angularaxis=dict(
                    tickfont=dict(color="white")
                )
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig_radar, use_container_width=True)


    # ==========================================
    # FULL PDF REPORT
    # ==========================================
    def generate_pdf():

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Mastery Score Prediction Report", ln=True, align="C")
        pdf.ln(5)

        pdf.cell(200, 10, txt=f"Predicted Score: {prediction:.2f}", ln=True)
        pdf.ln(5)

        pdf.cell(200, 10, txt="Learner Data:", ln=True)
        pdf.ln(3)

        for col in input_df.columns:
            value = input_df[col].values[0]
            pdf.cell(200, 8, txt=f"{col}: {value}", ln=True)

        temp_dir = tempfile.gettempdir()

        gauge_path = f"{temp_dir}/gauge.png"
        bar_path = f"{temp_dir}/bar.png"

        gauge.write_image(gauge_path)
        fig_bar.write_image(bar_path)

        pdf.add_page()
        pdf.cell(200, 10, txt="Prediction Gauge", ln=True)
        pdf.image(gauge_path, w=180)

        pdf.add_page()
        pdf.cell(200, 10, txt="Top 10 Feature Importance", ln=True)
        pdf.image(bar_path, w=180)

        if radar_data:
            radar_path = f"{temp_dir}/radar.png"
            fig_radar.write_image(radar_path)

            pdf.add_page()
            pdf.cell(200, 10, txt="Engagement Profile", ln=True)
            pdf.image(radar_path, w=180)

        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = generate_pdf()

    st.download_button(
        "📄 Download Full Report",
        pdf_bytes,
        "mastery_report.pdf",
        "application/pdf"
    )

# ==========================================
# TABS
# ==========================================
if st.session_state.page == "input":

    tab1, tab2 = st.tabs(["📝 Manual Entry", "📂 Dataset Entry"])

    # ==========================================
    # MANUAL ENTRY
    # ==========================================
    with tab1:

        st.subheader("Manual Entry (All Features)")

        # Initialize session state safely
        for feature in model_features:
            if feature not in st.session_state:
                if feature in numeric_cols:
                    st.session_state[feature] = 0.0
                else:
                    if feature in df_model.columns:
                        options = df_model[feature].dropna().unique().tolist()
                    else:
                        options = []

                    st.session_state[feature] = options[0] if options else ""

        # ==========================================
        # RANDOM GENERATE
        # ==========================================
        if st.button("🎲 Generate Random Data"):

            for col in numeric_cols:
                if col == "age":
                    st.session_state[col] = np.random.randint(15, 61)  # Age 15-60
                else:
                    st.session_state[col] = round(float(np.random.uniform(0, 100)), 2)

            for col in categorical_cols:
                if col in df_model.columns:
                    unique_vals = df_model[col].dropna().unique()
                else:
                    unique_vals = []

                if len(unique_vals) > 0:
                    st.session_state[col] = np.random.choice(unique_vals)

        # ==========================================
        # INPUT LAYOUT
        # ==========================================
        cols_per_row = 3
        rows = [model_features[i:i + cols_per_row]
                for i in range(0, len(model_features), cols_per_row)]

        manual_data = {}

        for row in rows:
            cols = st.columns(cols_per_row)

            for i, feature in enumerate(row):

                # ===============================
                # NUMERIC FEATURES
                # ===============================
                if feature in numeric_cols:

                    current_val = st.session_state.get(feature, 0.0)

                    try:
                        current_val = float(current_val)
                    except:
                        current_val = 0.0

                    # Restrict AGE between 15–60
                    if feature == "age":
                        manual_data[feature] = cols[i].number_input(
                            feature,
                            min_value=15,
                            max_value=60,
                            value=int(current_val) if 15 <= current_val <= 60 else 24,
                            step=1,
                            # key=feature
                        )
                    else:
                        manual_data[feature] = cols[i].number_input(
                            feature,
                            min_value=0.0,
                            value=current_val,
                            step=0.1,
                            key=feature
                        )

                # ===============================
                # CATEGORICAL FEATURES
                # ===============================
                else:
                    if feature in df_model.columns:
                        options = df_model[feature].dropna().unique().tolist()
                    else:
                        options = []

                    if options and st.session_state[feature] not in options:
                        st.session_state[feature] = options[0]

                    manual_data[feature] = cols[i].selectbox(
                        feature,
                        options=options,
                        key=feature
                    )

        manual_df = pd.DataFrame([manual_data])

        if st.button("🚀 Predict"):
            st.session_state.result_data = manual_df
            st.session_state.page = "result"
            st.rerun()


    # ==========================================
    # DATASET ENTRY
    # ==========================================
    with tab2:

        st.subheader("Search Learner ID")

        learner_input = st.text_input("Type Learner ID")

        suggestions = []
        if learner_input:
            suggestions = df[df["learner_id"].str.contains(
                learner_input.upper()
            )]["learner_id"].tolist()


        if suggestions:
            selected_learner = st.selectbox("Suggestions", suggestions)

            learner_data = df_model[df_model["learner_id"] == selected_learner]
            st.dataframe(learner_data)

            if st.button("🚀 Predict "):
                st.session_state.result_data = learner_data
                st.session_state.page = "result"
                st.rerun()

# ==========================================
# RESULT PAGE
# ==========================================
if st.session_state.page == "result":

    st.button("⬅ Back", on_click=lambda: st.session_state.update({"page": "input"}))

    show_results(st.session_state.result_data)
