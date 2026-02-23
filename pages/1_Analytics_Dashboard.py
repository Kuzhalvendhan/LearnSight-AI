import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def transparent_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig


st.set_page_config(page_title="Learning Analytics Dashboard", layout="wide")

st.markdown("""
<style>

/* Main Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
}

/* Optional: Make default container transparent */
.block-container {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/digital_learning_analytics_100k.csv")

df = load_data()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding:20px 10px;">
    <h2 style="margin-bottom:5px;">📊 DLI Dashboard</h2>
    <p style="font-size:14px; color:#cbd5e1;">
        Digital Learning Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation

# Filters
st.sidebar.markdown("### 🎛 Filters")

min_hours_input = st.sidebar.text_input(
    "Minimum Daily App Usage (Hours)",
    value="1",
    help="Enter hours between 0 and 24 (e.g. 1, 1.5, 0.75)"
)

# Safe conversion + validation
try:
    min_hours_val = float(min_hours_input)

    if 0 <= min_hours_val <= 24:
        min_hours = min_hours_val * 60
    else:
        st.sidebar.error("⚠ Please enter a value between 0 and 24 hours")
        min_hours = 0

except:
    st.sidebar.error("⚠ Invalid input. Use numbers like 1, 1.5, 0.75")
    min_hours = 0

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["country"].unique().tolist())
)

# -------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------
filtered_df = df.copy()
filtered_df = filtered_df[
    filtered_df["daily_app_minutes"] >= min_hours
]

if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["country"] == selected_country
    ]

# -------------------------------------------------------
# SIDEBAR STATS + DOWNLOAD
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.info(f"Filtered Learners: {len(filtered_df):,}")

if len(filtered_df) > 0:
    st.sidebar.info(
        f"Avg Mastery: {filtered_df['mastery_score'].mean():.2f}"
    )
else:
    st.sidebar.warning("No data after filtering.")

st.sidebar.markdown("---")

# Download Button
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv,
    file_name="filtered_learning_data.csv",
    mime="text/csv",
)

st.sidebar.markdown("---")
st.sidebar.success("System Active 🟢")

# -------------------------------------------------------
# HANDLE EMPTY DATA
# -------------------------------------------------------
if len(filtered_df) == 0:
    st.warning("No data matches your filters. Adjust filters.")
    st.stop()

st.markdown("## 🧭 Navigation")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔥 Engagement Analysis",
    "📈 Performance Insights",
    "🧠 Correlation Matrix"
])


# -------------------------------------------------------
# PAGE: OVERVIEW
# -------------------------------------------------------
with tab1:

    st.markdown("## 📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("👩‍🎓 Total Learners", f"{len(filtered_df):,}")
    col2.metric("⏳ Avg Daily App Minutes",
                f"{filtered_df['daily_app_minutes'].mean():.2f}")
    col3.metric("🎯 Avg Mastery Score",
                f"{filtered_df['mastery_score'].mean():.2f}")

    st.markdown("## 🌍 Learners by Country")

    country_counts = filtered_df["country"].value_counts().head(10)

    fig_country = px.pie(
        values=country_counts.values,
        names=country_counts.index,
    )

    fig_country = transparent_chart(fig_country)
    st.plotly_chart(fig_country, use_container_width=True)

# -------------------------------------------------------
# PAGE: ENGAGEMENT ANALYSIS
# -------------------------------------------------------
with tab2:

    st.markdown("## ⏳ Daily App Usage Distribution")

    fig_hours = px.histogram(
        filtered_df,
        x="daily_app_minutes",
        nbins=40,
    )

    fig_hours.update_traces(
        marker=dict(
            color="#00F5FF",
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        )
    )

    fig_hours.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.12)")
    )

    st.plotly_chart(fig_hours, use_container_width=True)

    st.markdown("## ⚡ Learning Efficiency vs Mastery")

    sample_df = filtered_df.sample(
        min(5000, len(filtered_df))
    )

    fig_efficiency = px.scatter(
        sample_df,
        x="learning_efficiency_score",
        y="mastery_score",
        opacity=0.5,
    )


    fig_efficiency.update_traces(
        marker=dict(
            size=6,
            color="#00F5FF",     # brighter electric cyan
            opacity=0.75,
            line=dict(width=0)
        )
    )

    fig_efficiency.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False
        )
    )


    st.plotly_chart(fig_efficiency, use_container_width=True)

# -------------------------------------------------------
# PAGE: PERFORMANCE INSIGHTS
# -------------------------------------------------------
with tab3:

    st.markdown("## 📝 Assignment Submission Rate vs Mastery")

    sample_df = filtered_df.sample(
        min(5000, len(filtered_df))
    )

    fig_scatter = px.scatter(
        sample_df,
        x="assignment_submission_rate",
        y="mastery_score",
        opacity=0.5,
    )

    fig_scatter.update_traces(
        marker=dict(
            size=6,
            color="#00F5FF",     # brighter electric cyan
            opacity=0.75,
            line=dict(width=0)
        )
    )

    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("## 💻 MOOC Platform Performance")

    platform_avg = filtered_df.groupby(
        "mooc_platform"
    )["mastery_score"].mean().reset_index()

    fig_platform = px.bar(
        platform_avg,
        x="mooc_platform",
        y="mastery_score",
    )

    fig_platform.update_traces(
        marker=dict(
            color="#00C6FF",
            line=dict(color="rgba(255,255,255,0.2)", width=1)
        )
    )

    fig_platform.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.12)")
    )

    st.plotly_chart(fig_platform, use_container_width=True)

# -------------------------------------------------------
# PAGE: CORRELATION MATRIX
# -------------------------------------------------------
with tab4:

    st.markdown("## 📈 Feature Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Blues"
        )
    )

    fig_heatmap = transparent_chart(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)
