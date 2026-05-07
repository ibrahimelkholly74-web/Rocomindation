# Hybrid Movie Recommendation System using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")

# -----------------------------
# Load data
# -----------------------------

BASE_DIR = Path(__file__).parent

movies = pd.read_csv(BASE_DIR / "movies.csv")
ratings = pd.read_csv(BASE_DIR / "ratings.csv")

with open(BASE_DIR / "cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open(BASE_DIR / "pred_matrix.pkl", "rb") as f:
    pred_matrix = pickle.load(f)

with open(BASE_DIR / "indices.pkl", "rb") as f:
    indices = pickle.load(f)

# -----------------------------
# Normalize function
# -----------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# -----------------------------
# Hybrid Model
# -----------------------------
def hybrid_recommend(user_id, title, alpha=0.7, top_n=10):

    # -----------------------------
    # Content-based scores
    # -----------------------------
    content_scores = normalize(cosine_sim[indices[title]])

    # -----------------------------
    # Cold Start handling
    # -----------------------------
    if user_id < 1 or user_id > pred_matrix.shape[0]:
        top_idx = np.argsort(content_scores)[::-1][:top_n]
        return movies.iloc[top_idx][['title', 'genres']]

    # -----------------------------
    # Collaborative scores
    # -----------------------------
    collab_scores = normalize(pred_matrix[user_id - 1])

    # -----------------------------
    # Alignment safety
    # -----------------------------
    min_len = min(len(content_scores), len(collab_scores))

    content_scores = content_scores[:min_len]
    collab_scores = collab_scores[:min_len]

    # -----------------------------
    # Weighted Hybrid
    # -----------------------------
    scores = (
        alpha * content_scores +
        (1 - alpha) * collab_scores
    )

    # -----------------------------
    # Top-N results
    # -----------------------------
    top_idx = np.argsort(scores)[::-1][:top_n]

    return movies.iloc[top_idx][['title', 'genres']]

# -----------------------------
# UI Inputs
# -----------------------------
max_user = pred_matrix.shape[0]

st.title("Hybrid Movie Recommendation System")

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
user_id = st.number_input(
    "User ID (leave default for testing)",
    min_value=1,
    value=1
)

movie_list = movies['title'].values
selected_movie = st.selectbox("Select Movie", movie_list)

st.markdown("---")

# -----------------------------
# Model Balance
# -----------------------------
st.markdown("### Model Contribution")

alpha = st.slider("Balance (Content vs Collaborative)", 0.0, 1.0, 0.7)

st.progress(alpha)

st.caption(f"Content-Based: {alpha:.2f} | Collaborative: {1-alpha:.2f}")

st.markdown("---")

# -----------------------------
# Recommendation Depth
# -----------------------------
st.markdown("### Recommendation Depth")

top_n = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

st.caption(f"Showing {top_n} recommended movies")

st.markdown("---")

# -----------------------------
# Button
# -----------------------------
if st.button("Get Recommendations"):

    results = hybrid_recommend(user_id, selected_movie, alpha, top_n)

    st.subheader("Recommendations")

    for i, row in enumerate(results.itertuples(), start=1):
        st.markdown(f"**{i}. {row.title}**")
        st.caption(row.genres)
        st.markdown("---")
