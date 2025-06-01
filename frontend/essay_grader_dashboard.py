# essay_grader_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px

st.set_page_config(
    page_title="Essay Grader Evaluation Dashboard",
    layout="wide",
    page_icon="📝"
)

@st.cache_data
def load_data():
    return pd.read_csv("results/model_with_fine_tuning_results.csv")

df = load_data()

st.sidebar.header("Filters")
score_filter = st.sidebar.multiselect(
    "Filter by true score",
    options=sorted(df['true_score'].unique()),
    default=sorted(df['true_score'].unique())
)

correct_filter = st.sidebar.multiselect(
    "Filter by correctness",
    options=["Correct", "Incorrect"],
    default=["Correct", "Incorrect"]
)

filtered_df = df[
    (df['true_score'].isin(score_filter)) & 
    (df['correct'].isin([x == "Correct" for x in correct_filter]))
]

st.title("Essay Grader Evaluation Dashboard")
st.write(f"Evaluating {len(filtered_df)} of {len(df)} samples")

col1, col2, col3 = st.columns(3)
with col1:
    accuracy = (df['correct'].sum() / len(df)) * 100
    st.metric("Overall Accuracy", f"{accuracy:.1f}%")
with col2:
    avg_error = (df['true_score'] - df['pred_score']).abs().mean()
    st.metric("Average Error", f"{avg_error:.2f}")
with col3:
    correct_counts = df['correct'].value_counts()
    st.metric("Correct/Incorrect", f"{correct_counts[True]} / {correct_counts[False]}")

tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Error Analysis", "Sample Responses"])

with tab1:
    st.header("Performance Metrics")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['true_score'], df['pred_score'], labels=[1,2,3,4])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
    
    st.subheader("Score Distribution")
    fig = px.histogram(df, x="true_score", color="correct", 
                      barmode="group", nbins=4,
                      title="True Score Distribution by Correctness")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Error Analysis")
    
    st.subheader("Error Distribution")
    df['error'] = (df['true_score'] - df['pred_score']).abs()
    fig = px.histogram(df, x="error", nbins=4,
                      title="Absolute Error Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Largest Errors")
    largest_errors = df.nlargest(5, 'error')
    st.dataframe(largest_errors[['question', 'true_score', 'pred_score', 'error']])

with tab3:
    st.header("Sample Responses")
    
    st.subheader("Filtered Responses")
    for _, row in filtered_df.iterrows():
        with st.expander(f"Question: {row['question']} (True: {row['true_score']}, Predicted: {row['pred_score']})"):
            st.write(f"**Student Answer:** {row['student_answer']}")
            st.write(f"**Model Response:** {row['full_response']}")
            st.write(f"**Correct:** {'✅' if row['correct'] else '❌'}")