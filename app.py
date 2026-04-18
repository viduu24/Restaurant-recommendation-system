# ============================================================
#  Restaurant Recommendation System — Streamlit Dashboard
#  app.py  (main entry point)
# ============================================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import streamlit as st

from tabs.tab_overview       import render_overview
from tabs.tab_exploration    import render_exploration
from tabs.tab_recommendation import render_recommendation

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2.4rem; font-weight:700; color:#E8450A; margin-bottom:0.2rem; }
    .sub-header  { font-size:1rem;   color:#888;      margin-bottom:1.5rem; }
    .metric-card {
        background: linear-gradient(135deg,#1e1e2e 0%,#2a2a3e 100%);
        border:1px solid #3a3a5c; border-radius:12px;
        padding:1.2rem 1.5rem; text-align:center; margin-bottom:0.5rem;
    }
    .metric-value { font-size:2rem; font-weight:700; color:#E8450A; }
    .metric-label { font-size:0.85rem; color:#aaa; margin-top:0.2rem; }
    .rec-card {
        background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
        border:1px solid #E8450A44; border-radius:14px;
        padding:1.2rem 1.5rem; margin-bottom:1rem;
    }
    .rec-rank  { font-size:1.8rem; font-weight:800; color:#E8450A; }
    .rec-name  { font-size:1.2rem; font-weight:600; color:#fff; }
    .rec-meta  { font-size:0.85rem; color:#aaa; margin-top:0.3rem; }
    .rec-badge {
        display:inline-block; background:#E8450A22;
        border:1px solid #E8450A66; border-radius:20px;
        padding:2px 10px; font-size:0.78rem; color:#E8450A;
        margin-right:6px; margin-top:6px;
    }
    .section-title {
        font-size:1.3rem; font-weight:600; color:#E8450A;
        border-left:4px solid #E8450A; padding-left:0.6rem;
        margin:1.5rem 0 1rem;
    }
    section[data-testid="stSidebar"] { background:#0f0f1a; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading restaurant dataset…")
def load_business_data():
    df = pd.read_csv("data/business_reviews_agg.csv")
    df["IS_OPEN_LABEL"] = df["IS_OPEN"].map({1: "Open", 0: "Closed"})
    df["primary_category"] = df["CATEGORIES"].str.split(",").str[0].str.strip()
    return df


def _ensure_surprise():
    try:
        import surprise  # noqa
        return True
    except ModuleNotFoundError:
        import subprocess, sys
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "scikit-surprise", "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import surprise  # noqa
            return True
        except Exception:
            return False


@st.cache_resource(show_spinner="Loading SVD model…")
def load_svd_model():
    path = os.path.join(BASE_DIR, "data", "svd_model.pkl")
    if not os.path.exists(path):
        return None
    if not _ensure_surprise():
        st.sidebar.warning("⚠️ scikit-surprise could not be installed. Run `pip install scikit-surprise`.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading KNN model…")
def load_knn_model():
    path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
    if not os.path.exists(path):
        return None
    if not _ensure_surprise():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading encoders…")
def load_encoders():
    data_dir  = os.path.join(BASE_DIR, "data")
    user_path = os.path.join(data_dir, "user_encoder.pkl")
    biz_path  = os.path.join(data_dir, "business_encoder.pkl")
    if not (os.path.exists(user_path) and os.path.exists(biz_path)):
        return None, None
    with open(user_path, "rb") as f:
        user_enc = pickle.load(f)
    with open(biz_path, "rb") as f:
        biz_enc  = pickle.load(f)
    return user_enc, biz_enc


@st.cache_data(show_spinner="Loading ratings data…")
def load_ratings():
    path = os.path.join(BASE_DIR, "data", "ratings_train.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading business metadata…")
def load_business_meta():
    path = os.path.join(BASE_DIR, "data", "business_meta.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def connect_pinecone():
    if "pinecone_index" in st.session_state and "embedder" in st.session_state:
        return st.session_state["pinecone_index"], st.session_state["embedder"]
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        PINECONE_API_KEY = (
            st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY", "")
        )
        if not PINECONE_API_KEY:
            st.warning("Pinecone API key not set.")
            return None, None

        pc       = Pinecone(api_key=PINECONE_API_KEY)
        pine_idx = pc.Index("811-business-description")
        embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        st.session_state["pinecone_index"] = pine_idx
        st.session_state["embedder"]       = embedder
        return pine_idx, embedder
    except Exception:
        st.session_state["pinecone_index"] = None
        st.session_state["embedder"]       = None
        return None, None


# ════════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def normalize_scores(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 1.0
    return (series - mn) / (mx - mn)


def pinecone_search(query_text, city=None, state=None, top_k=10):
    pine_idx, embedder = connect_pinecone()
    if pine_idx is None or embedder is None:
        return pd.DataFrame()

    vec = embedder.encode([query_text], normalize_embeddings=True)[0].tolist()

    pine_filter = {}
    if city and state:
        pine_filter = {"$and": [{"city": {"$eq": city}}, {"state": {"$eq": state.upper()}}]}
    elif city:
        pine_filter = {"city": {"$eq": city}}
    elif state:
        pine_filter = {"state": {"$eq": state.upper()}}

    results = pine_idx.query(
        vector=vec, top_k=top_k, include_metadata=True,
        filter=pine_filter if pine_filter else None
    )

    rows = []
    for m in results.get("matches", []):
        md = m.get("metadata", {})
        rows.append({
            "business_id":      m["id"],
            "business_name":    md.get("business_name", "Unknown"),
            "city":             md.get("city", ""),
            "state":            md.get("state", ""),
            "primary_category": md.get("primary_category", ""),
            "avg_stars":        md.get("avg_stars", 0.0),
            "pine_score":       m.get("score", 0.0),
            "description":      md.get("description", ""),
        })
    return pd.DataFrame(rows)


def hybrid_recommend_existing(user_id, svd_model, knn_model, user_encoder,
                               biz_encoder, train_df, business_meta,
                               city=None, state=None, top_k=5,
                               preference_text=None,
                               svd_weight=0.6, knn_weight=0.4):
    rated_biz  = set(train_df[train_df["user_id"] == user_id]["business_id"].tolist())
    candidates = business_meta.copy()
    if city:
        candidates = candidates[candidates["city"].str.lower() == city.lower()]
    if state:
        candidates = candidates[candidates["state"].str.upper() == state.upper()]
    candidates = candidates[~candidates["business_id"].isin(rated_biz)].reset_index(drop=True)
    if candidates.empty:
        return pd.DataFrame()

    svd_preds = [svd_model.predict(user_id, b).est for b in candidates["business_id"]]
    candidates["svd_score"] = svd_preds
    candidates["svd_norm"]  = normalize_scores(candidates["svd_score"])

    if knn_model is not None:
        knn_preds = [knn_model.predict(user_id, b).est for b in candidates["business_id"]]
        candidates["knn_score"] = knn_preds
        candidates["knn_norm"]  = normalize_scores(candidates["knn_score"])
        candidates["cf_score"]  = svd_weight * candidates["svd_norm"] + knn_weight * candidates["knn_norm"]
    else:
        candidates["cf_score"] = candidates["svd_norm"]

    if preference_text:
        pine_df = pinecone_search(preference_text, city=city, state=state, top_k=50)
        if not pine_df.empty:
            pine_map = pine_df.set_index("business_id")["pine_score"].to_dict()
            candidates["pine_score"] = candidates["business_id"].map(pine_map).fillna(0.0)
            candidates["pine_norm"]  = normalize_scores(candidates["pine_score"])
            candidates["final_score"] = 0.7 * candidates["cf_score"] + 0.3 * candidates["pine_norm"]
        else:
            candidates["final_score"] = candidates["cf_score"]
    else:
        candidates["final_score"] = candidates["cf_score"]

    return candidates.nlargest(top_k, "final_score").reset_index(drop=True)


def recommend_new_user(preference_text, city=None, state=None, top_k=5):
    df = pinecone_search(preference_text, city=city, state=state, top_k=top_k)
    if df.empty:
        return df
    df = df.rename(columns={"avg_stars": "business_avg_stars"})
    df["final_score"] = df["pine_score"]
    return df.head(top_k).reset_index(drop=True)


def master_recommend(user_id, preference_text, city, state, top_k,
                     svd_model, knn_model, user_encoder,
                     biz_encoder, train_df, business_meta,
                     svd_weight=0.6, knn_weight=0.4):
    models_available = (
        svd_model is not None
        and user_encoder is not None
        and not train_df.empty
        and not business_meta.empty
    )
    is_existing = (
        models_available
        and user_id
        and user_id in user_encoder.get("str2idx", {})
    )

    if is_existing:
        results = hybrid_recommend_existing(
            user_id=user_id, svd_model=svd_model, knn_model=knn_model,
            user_encoder=user_encoder, biz_encoder=biz_encoder,
            train_df=train_df, business_meta=business_meta,
            city=city or None, state=state or None, top_k=top_k,
            preference_text=preference_text or None,
            svd_weight=svd_weight, knn_weight=knn_weight,
        )
        strategy = "Hybrid (SVD + KNN" + (" + Pinecone)" if preference_text else ")")
    else:
        if not preference_text:
            if not models_available:
                return pd.DataFrame(), "SVD model not loaded. Enter a preference description to use Pinecone search."
            return pd.DataFrame(), "No preference text provided for new user"

        results = recommend_new_user(preference_text, city=city or None, state=state or None, top_k=top_k)

        if not models_available:
            strategy = "Semantic Search / Pinecone only"
        elif user_id and user_encoder and user_id not in user_encoder.get("str2idx", {}):
            strategy = "Semantic Search / Pinecone (User ID not found)"
        else:
            strategy = "Semantic Search / Pinecone (New User)"

    return results, strategy


# ════════════════════════════════════════════════════════════════════════════════
#  LOAD DATA & MODELS
# ════════════════════════════════════════════════════════════════════════════════

df                = load_business_data()
svd_model         = load_svd_model()
knn_model         = load_knn_model()
user_enc, biz_enc = load_encoders()
train_df          = load_ratings()
biz_meta          = load_business_meta()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Recommender")
    st.markdown("---")
    st.markdown("### 📊 Dataset At a Glance")
    st.metric("Total Restaurants", f"{len(df):,}")
    st.metric("Unique Cities",      f"{df['CITY'].nunique():,}")
    st.metric("Average Rating",     f"{df['BUSINESS_AVG_STARS'].mean():.2f} ⭐")
    st.metric("Open Restaurants",   f"{df['IS_OPEN'].sum():,}")

    st.markdown("---")
    st.markdown("### 🤖 Model Status")

    def _status(obj, label, hint=""):
        if obj is not None:
            st.markdown(f"✅ **{label}**")
        else:
            st.markdown(f"❌ **{label}**")
            if hint:
                st.caption(hint)

    _status(svd_model, "SVD Model",        "Run: `pip install scikit-surprise`" if svd_model is None else "")
    if knn_model is not None:
        st.markdown("✅ **KNN Model**")
    else:
        st.markdown("⚠️ **KNN Model** *(optional)*")
        st.caption("Run notebook_4 to enable full hybrid")
    _status(user_enc,  "User Encoder",     "Run notebook_2_data_prep first" if user_enc is None else "")
    _status(biz_enc,   "Business Encoder", "Run notebook_2_data_prep first" if biz_enc  is None else "")
    st.markdown("✅ **Pinecone** *(lazy-loaded on first use)*")
    st.markdown("---")
    st.caption("Built with Streamlit · Pinecone · Surprise")


# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🍽️ Restaurant Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hybrid SVD + KNN + Pinecone semantic search engine</p>', unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_explore, tab_recommend = st.tabs(
    ["📈 Overview", "🔍 Exploration", "🤖 Recommendation"]
)

with tab_overview:
    render_overview(df)

with tab_explore:
    render_exploration(df)

with tab_recommend:
    render_recommendation(df, svd_model, knn_model, user_enc, biz_enc,
                          train_df, biz_meta, master_recommend)
