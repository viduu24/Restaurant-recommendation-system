# ============================================================
#  Restaurant Recommendation System — Streamlit Dashboard
#  app.py
#
#  Structure:
#   ├── Shared helpers  (data loading, model loading)
#   ├── Tab 1: Overview       — dataset KPIs & summary
#   ├── Tab 2: Exploration    — interactive charts & filters
#   └── Tab 3: Recommendation — hybrid SVD+KNN+Pinecone engine
#
#  Run: streamlit run app.py
# ============================================================

import os
import re
import pickle
import warnings
import snowflake.connector
warnings.filterwarnings("ignore")

# Base directory — resolved relative to this file so it works locally
# AND on Streamlit Cloud (repo root) without any path changes needed.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — clean dark-accent theme ────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #E8450A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    /* KPI metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #E8450A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    /* Restaurant recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #E8450A44;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .rec-rank {
        font-size: 1.8rem;
        font-weight: 800;
        color: #E8450A;
    }
    .rec-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
    }
    .rec-meta {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .rec-badge {
        display: inline-block;
        background: #E8450A22;
        border: 1px solid #E8450A66;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #E8450A;
        margin-right: 6px;
        margin-top: 6px;
    }
    /* Section divider */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E8450A;
        border-left: 4px solid #E8450A;
        padding-left: 0.6rem;
        margin: 1.5rem 0 1rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f0f1a;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING  (cached so they load only once per session)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading restaurant dataset…")


@st.cache_data
def load_business_data():
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"]
    )
    df = pd.read_sql("SELECT * FROM USER_REVIEWS_AGG", conn)
    conn.close()
    return df


def _ensure_surprise():
    """
    Ensure scikit-surprise is importable.
    The SVD/KNN pickle files were saved with scikit-surprise, so the library
    must be present before unpickling.  If it isn't installed we attempt a
    silent pip install so the user doesn't have to do anything manually.
    Returns True if surprise is available, False otherwise.
    """
    try:
        import surprise  # noqa: F401 — just checking availability
        return True
    except ModuleNotFoundError:
        import subprocess, sys
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "scikit-surprise", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import surprise  # noqa: F401 — re-try after install
            return True
        except Exception:
            return False


@st.cache_resource(show_spinner="Loading SVD model…")
def load_svd_model():
    """
    Load the pre-trained SVD collaborative-filtering model from disk.
    scikit-surprise must be available because the pickle was created with it.
    Returns None (and shows a sidebar warning) if the library is missing or
    the file doesn't exist.
    """
    svd_path = os.path.join(BASE_DIR, "models", "svd_model.pkl")
    if not os.path.exists(svd_path):
        return None  # File simply hasn't been generated yet

    if not _ensure_surprise():
        st.sidebar.warning(
            "⚠️ `scikit-surprise` could not be installed automatically.  "
            "Run `pip install scikit-surprise` in your terminal, then refresh."
        )
        return None

    with open(svd_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading KNN model…")
def load_knn_model():
    """
    Load the pre-trained KNN collaborative-filtering model from disk.
    Same scikit-surprise dependency as the SVD model.
    Returns None if the library is missing or the file doesn't exist.
    """
    knn_path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
    if not os.path.exists(knn_path):
        return None  # File simply hasn't been generated yet

    if not _ensure_surprise():
        # Warning already shown by load_svd_model; no need to repeat
        return None

    with open(knn_path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading encoders…")
def load_encoders():
    """
    Load user & business ID encoders produced during data-prep.
    Returns (user_encoder, biz_encoder) dicts or (None, None) if missing.
    Each encoder has keys 'str2idx' and 'idx2str'.
    """
    data_dir = os.path.join(BASE_DIR, "data")
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
    """Load the training ratings CSV (user_id, business_id, target_rating)."""
    path = os.path.join(BASE_DIR, "data", "ratings_train.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading business metadata…")
def load_business_meta():
    """Load the slim business metadata CSV used by the SVD/KNN models."""
    path = os.path.join(BASE_DIR, "data", "business_meta.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def connect_pinecone():
    """
    Connect to Pinecone and return (index, embedder) tuple.
    Returns (None, None) on failure.
    Uses st.session_state to avoid re-creating the connection each rerun.
    """
    if "pinecone_index" in st.session_state and "embedder" in st.session_state:
        return st.session_state["pinecone_index"], st.session_state["embedder"]

    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        # Load Pinecone API key from Streamlit secrets (preferred) or env var.
        # On Streamlit Cloud: add PINECONE_API_KEY in App Settings → Secrets.
        # Locally: set it in .streamlit/secrets.toml or as an environment variable.
        PINECONE_API_KEY = (
            st.secrets.get("PINECONE_API_KEY")
            or os.getenv("PINECONE_API_KEY", "")
        )
        if not PINECONE_API_KEY:
            st.warning("Pinecone API key not set. Add it to .streamlit/secrets.toml or Streamlit Cloud secrets.")
            return None, None

        PINECONE_INDEX_NAME = "811-business-description"
        EMBEDDING_MODEL     = "sentence-transformers/all-mpnet-base-v2"

        pc        = Pinecone(api_key=PINECONE_API_KEY)
        pine_idx  = pc.Index(PINECONE_INDEX_NAME)
        embedder  = SentenceTransformer(EMBEDDING_MODEL)

        st.session_state["pinecone_index"] = pine_idx
        st.session_state["embedder"]       = embedder
        return pine_idx, embedder

    except Exception as e:
        # Pinecone or sentence-transformers may not be installed in demo mode
        st.session_state["pinecone_index"] = None
        st.session_state["embedder"]       = None
        return None, None


# ════════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def normalize_scores(series: pd.Series) -> pd.Series:
    """Min-max normalise a score series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 1.0
    return (series - mn) / (mx - mn)


def pinecone_search(query_text, city=None, state=None, top_k=10):
    """
    Perform semantic (vector) search in Pinecone.

    Args:
        query_text : Free-text preference description.
        city       : Optional city filter applied server-side.
        state      : Optional state filter applied server-side.
        top_k      : Number of results to retrieve.

    Returns:
        pd.DataFrame with columns [business_id, business_name, city, state,
                                   primary_category, avg_stars, pine_score].
        Returns an empty DataFrame if Pinecone is unavailable.
    """
    pine_idx, embedder = connect_pinecone()
    if pine_idx is None or embedder is None:
        return pd.DataFrame()

    # Encode query as a unit-normalised 768-dim vector
    vec = embedder.encode([query_text], normalize_embeddings=True)[0].tolist()

    # Build optional metadata filter for location
    pine_filter = {}
    if city and state:
        pine_filter = {"$and": [{"city": {"$eq": city}}, {"state": {"$eq": state.upper()}}]}
    elif city:
        pine_filter = {"city": {"$eq": city}}
    elif state:
        pine_filter = {"state": {"$eq": state.upper()}}

    results = pine_idx.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
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
    """
    Hybrid recommendation for an EXISTING user (has rating history).

    Strategy:
      1. Predict ratings for all unrated restaurants with SVD (always used).
      2. If KNN model is also available, blend SVD + KNN scores with
         configurable weights.  If KNN is None, SVD-only scoring is used
         (knn_model.pkl not yet generated — run notebook_4 to enable it).
      3. If preference_text is given, also query Pinecone and boost
         the blended score with cosine similarity (70% CF + 30% semantic).
      4. Return top-K results.

    Args:
        user_id        : User string ID known to the SVD model.
        svd_model      : Loaded Surprise SVD instance (required).
        knn_model      : Loaded Surprise KNNWithMeans instance (optional —
                         pass None to use SVD-only scoring).
        user_encoder   : Dict with 'str2idx' / 'idx2str' mappings.
        biz_encoder    : Dict with 'str2idx' / 'idx2str' mappings.
        train_df       : Training ratings DataFrame.
        business_meta  : Business metadata DataFrame.
        city / state   : Optional location filters.
        top_k          : Number of recommendations.
        preference_text: Optional free-text to enable Pinecone boost.
        svd_weight     : Weight for SVD score (default 0.6).
        knn_weight     : Weight for KNN score (default 0.4).

    Returns:
        pd.DataFrame of top-K recommendations with a 'final_score' column.
    """
    # Restaurants already rated by this user — exclude them from candidates
    rated_biz = set(train_df[train_df["user_id"] == user_id]["business_id"].tolist())

    # Start from the full business pool, apply optional location filter
    candidates = business_meta.copy()
    if city:
        candidates = candidates[candidates["city"].str.lower() == city.lower()]
    if state:
        candidates = candidates[candidates["state"].str.upper() == state.upper()]

    # Remove already-rated restaurants
    candidates = candidates[~candidates["business_id"].isin(rated_biz)].reset_index(drop=True)

    if candidates.empty:
        return pd.DataFrame()

    # ── Score each candidate ─────────────────────────────────────────────────
    # SVD is always available; KNN is optional (may be None if not yet trained)
    svd_preds = [svd_model.predict(user_id, biz_id).est
                 for biz_id in candidates["business_id"]]
    candidates["svd_score"] = svd_preds
    candidates["svd_norm"]  = normalize_scores(candidates["svd_score"])

    if knn_model is not None:
        # Full hybrid: blend SVD + KNN
        knn_preds = [knn_model.predict(user_id, biz_id).est
                     for biz_id in candidates["business_id"]]
        candidates["knn_score"] = knn_preds
        candidates["knn_norm"]  = normalize_scores(candidates["knn_score"])
        candidates["cf_score"]  = (
            svd_weight * candidates["svd_norm"] +
            knn_weight * candidates["knn_norm"]
        )
    else:
        # SVD-only fallback: KNN model not yet available
        candidates["cf_score"] = candidates["svd_norm"]

    # ── Optional Pinecone semantic boost ─────────────────────────────────────
    if preference_text:
        pine_df = pinecone_search(preference_text, city=city, state=state, top_k=50)
        if not pine_df.empty:
            pine_map = pine_df.set_index("business_id")["pine_score"].to_dict()
            candidates["pine_score"] = candidates["business_id"].map(pine_map).fillna(0.0)
            candidates["pine_norm"]  = normalize_scores(candidates["pine_score"])
            # 70% CF score + 30% Pinecone semantic similarity
            candidates["final_score"] = 0.7 * candidates["cf_score"] + 0.3 * candidates["pine_norm"]
        else:
            candidates["final_score"] = candidates["cf_score"]
    else:
        candidates["final_score"] = candidates["cf_score"]

    return candidates.nlargest(top_k, "final_score").reset_index(drop=True)


def recommend_new_user(preference_text, city=None, state=None, top_k=5):
    """
    Recommendation for a NEW user (no rating history).
    Falls back entirely to Pinecone semantic search.

    Args:
        preference_text: What the user is looking for.
        city / state   : Optional location filters.
        top_k          : Number of results.

    Returns:
        pd.DataFrame (or empty DataFrame if Pinecone unavailable).
    """
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
    """
    Master dispatcher — routes to the correct recommendation strategy:
      * Existing user + models loaded  -> Hybrid SVD + KNN (+ optional Pinecone boost)
      * Existing user + models missing -> Pinecone semantic search (graceful fallback)
      * New / unknown user             -> Pinecone semantic search only

    If SVD/KNN models are None (e.g. scikit-surprise not installed),
    the function automatically falls back to Pinecone so the Recommendation
    tab still works without the collaborative-filtering models.

    Returns a (DataFrame, strategy_label) tuple.
    """
    # SVD is the required model; KNN is optional (enhances but not required)
    # The app works in SVD-only mode if knn_model.pkl has not been generated yet
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
        # Full hybrid path: SVD + KNN + optional Pinecone boost
        results = hybrid_recommend_existing(
            user_id=user_id,
            svd_model=svd_model,
            knn_model=knn_model,
            user_encoder=user_encoder,
            biz_encoder=biz_encoder,
            train_df=train_df,
            business_meta=business_meta,
            city=city or None,
            state=state or None,
            top_k=top_k,
            preference_text=preference_text or None,
            svd_weight=svd_weight,
            knn_weight=knn_weight,
        )
        strategy = "Hybrid (SVD + KNN" + (" + Pinecone)" if preference_text else ")")

    else:
        # Pinecone-only path: new user OR models not yet installed
        if not preference_text:
            if not models_available:
                return (
                    pd.DataFrame(),
                    "SVD model not loaded. Enter a preference description "
                    "to use Pinecone search, or run: conda install -c conda-forge scikit-surprise"
                )
            return pd.DataFrame(), "No preference text provided for new user"

        results = recommend_new_user(
            preference_text=preference_text,
            city=city or None,
            state=state or None,
            top_k=top_k,
        )

        if not models_available:
            strategy = "Semantic Search / Pinecone only (install scikit-surprise for hybrid)"
        elif user_id and user_encoder and user_id not in user_encoder.get("str2idx", {}):
            strategy = "Semantic Search / Pinecone (User ID not found in training data)"
        else:
            strategy = "Semantic Search / Pinecone (New User)"

    return results, strategy


# ════════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def star_html(rating):
    """Return a ★/☆ HTML string for a 0–5 star rating."""
    full  = int(round(rating))
    stars = "★" * full + "☆" * (5 - full)
    return f'<span style="color:#FFD700;font-size:1.1rem">{stars}</span> {rating:.1f}'


def render_rec_card(rank, row, has_description=False):
    """Render a single recommendation result as a styled HTML card."""
    name     = row.get("business_name", row.get("BUSINESS_NAME", "N/A"))
    city     = row.get("city",  row.get("CITY",  ""))
    state    = row.get("state", row.get("STATE", ""))
    cat      = row.get("primary_category", row.get("categories", ""))
    stars    = float(row.get("business_avg_stars", row.get("avg_stars", 0)))
    score    = row.get("final_score", None)
    desc     = row.get("description", "") if has_description else ""

    score_badge = (
        f'<span class="rec-badge">Score: {score:.3f}</span>' if score is not None else ""
    )
    desc_html = f'<p style="color:#ccc;font-size:0.88rem;margin-top:0.5rem">{desc}</p>' if desc else ""

    st.markdown(f"""
    <div class="rec-card">
        <span class="rec-rank">#{rank}</span>
        <span class="rec-name" style="margin-left:10px">{name}</span>
        <div class="rec-meta">📍 {city}, {state} &nbsp;|&nbsp; {star_html(stars)}</div>
        <div style="margin-top:6px">
            <span class="rec-badge">{cat}</span>
            {score_badge}
        </div>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  LOAD EVERYTHING BEFORE RENDERING TABS
# ════════════════════════════════════════════════════════════════════════════════

df          = load_business_data()
svd_model   = load_svd_model()
knn_model   = load_knn_model()
user_enc, biz_enc = load_encoders()
train_df    = load_ratings()
biz_meta    = load_business_meta()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Recommender")
    st.markdown("---")

    st.markdown("### 📊 Dataset At a Glance")
    st.metric("Total Restaurants",   f"{len(df):,}")
    st.metric("Unique Cities",        f"{df['CITY'].nunique():,}")
    st.metric("Average Rating",       f"{df['BUSINESS_AVG_STARS'].mean():.2f} ⭐")
    st.metric("Open Restaurants",     f"{df['IS_OPEN'].sum():,}")

    st.markdown("---")
    st.markdown("### 🤖 Model Status")

    def status(obj, label, hint=""):
        if obj is not None:
            st.markdown(f"✅ **{label}**")
        else:
            st.markdown(f"❌ **{label}**")
            if hint:
                st.caption(hint)

    _surprise_hint = "Run: `conda install -c conda-forge scikit-surprise`"
    _data_hint     = "Run notebook_2_data_prep first"
    _knn_hint      = "Optional — run notebook_4 to enable full hybrid"

    # SVD is required; KNN is optional (improves results but not needed)
    status(svd_model, "SVD Model",        _surprise_hint if svd_model is None else "")

    # KNN: show a softer "optional" warning instead of a hard error
    if knn_model is not None:
        st.markdown("✅ **KNN Model**")
    else:
        st.markdown("⚠️ **KNN Model** *(optional)*")
        st.caption(_knn_hint)

    status(user_enc,  "User Encoder",     _data_hint if user_enc is None else "")
    status(biz_enc,   "Business Encoder", _data_hint if biz_enc  is None else "")
    st.markdown("✅ **Pinecone** *(lazy-loaded on first use)*")

    st.markdown("---")
    st.caption("Built with Streamlit · Pinecone · Surprise")


# ── Page title ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🍽️ Restaurant Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hybrid SVD + KNN + Pinecone semantic search engine</p>', unsafe_allow_html=True)

# ── Three main tabs ──────────────────────────────────────────────────────────────
tab_overview, tab_explore, tab_recommend = st.tabs(
    ["📈 Overview", "🔍 Exploration", "🤖 Recommendation"]
)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
#  Shows high-level KPIs, distribution charts, and a quick geographic snapshot.
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # ── Section: KPI Row ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Key Metrics</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    kpi_data = [
        (c1, f"{len(df):,}",                               "Total Restaurants"),
        (c2, f"{df['CITY'].nunique():,}",                  "Unique Cities"),
        (c3, f"{df['STATE'].nunique():,}",                 "States / Provinces"),
        (c4, f"{df['BUSINESS_AVG_STARS'].mean():.2f} ⭐",  "Avg Star Rating"),
        (c5, f"{df['IS_OPEN'].mean()*100:.1f}%",           "Currently Open"),
    ]

    for col, val, lbl in kpi_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section: Rating & Open/Closed distribution ────────────────────────────
    st.markdown('<p class="section-title">Rating & Status Distribution</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Histogram of average star ratings
        fig_rating = px.histogram(
            df, x="BUSINESS_AVG_STARS",
            nbins=18,
            title="Distribution of Average Star Ratings",
            labels={"BUSINESS_AVG_STARS": "Average Stars"},
            color_discrete_sequence=["#E8450A"],
        )
        fig_rating.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.05,
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with col_b:
        # Pie chart: Open vs Closed
        status_counts = df["IS_OPEN_LABEL"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_status = px.pie(
            status_counts, names="Status", values="Count",
            title="Open vs Closed Restaurants",
            color_discrete_sequence=["#E8450A", "#444"],
            hole=0.4,
        )
        fig_status.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # ── Section: Top States & Categories ──────────────────────────────────────
    st.markdown('<p class="section-title">Top States & Categories</p>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Bar chart: top 10 states by restaurant count
        top_states = (
            df["STATE"].value_counts().head(10).reset_index()
        )
        top_states.columns = ["State", "Count"]
        fig_states = px.bar(
            top_states, x="Count", y="State",
            orientation="h",
            title="Top 10 States by Restaurant Count",
            color="Count",
            color_continuous_scale=["#2a2a3e", "#E8450A"],
        )
        fig_states.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_states, use_container_width=True)

    with col_d:
        # Bar chart: top 10 primary categories
        top_cats = (
            df["primary_category"].value_counts().head(10).reset_index()
        )
        top_cats.columns = ["Category", "Count"]
        fig_cats = px.bar(
            top_cats, x="Count", y="Category",
            orientation="h",
            title="Top 10 Primary Categories",
            color="Count",
            color_continuous_scale=["#2a2a3e", "#E8450A"],
        )
        fig_cats.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_cats, use_container_width=True)

    # ── Section: Geographic Map ────────────────────────────────────────────────
    st.markdown('<p class="section-title">Geographic Distribution</p>', unsafe_allow_html=True)

    # Sample up to 3000 points so the map renders quickly
    map_df = df[["LATITUDE", "LONGITUDE", "BUSINESS_NAME",
                 "BUSINESS_AVG_STARS", "primary_category"]].dropna().sample(
        min(3000, len(df)), random_state=42
    )

    fig_map = px.scatter_mapbox(
        map_df,
        lat="LATITUDE", lon="LONGITUDE",
        hover_name="BUSINESS_NAME",
        hover_data={"BUSINESS_AVG_STARS": True, "primary_category": True,
                    "LATITUDE": False, "LONGITUDE": False},
        color="BUSINESS_AVG_STARS",
        color_continuous_scale=["#444", "#E8450A"],
        size_max=8,
        zoom=3,
        title="Restaurant Locations (sample of 3,000)",
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        coloraxis_colorbar=dict(title="Stars"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Section: Review Engagement Metrics ────────────────────────────────────
    st.markdown('<p class="section-title">Review Engagement</p>', unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    with col_e:
        # Scatter: review count vs average rating
        scatter_df = df[["BUSINESS_REVIEW_COUNT", "BUSINESS_AVG_STARS",
                          "primary_category"]].dropna().sample(
            min(2000, len(df)), random_state=1
        )
        fig_scatter = px.scatter(
            scatter_df,
            x="BUSINESS_REVIEW_COUNT",
            y="BUSINESS_AVG_STARS",
            color="primary_category",
            title="Review Count vs Average Rating",
            labels={"BUSINESS_REVIEW_COUNT": "Review Count",
                    "BUSINESS_AVG_STARS": "Avg Stars"},
            opacity=0.6,
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_f:
        # Box plot: star ratings per top-8 categories
        top8 = df["primary_category"].value_counts().head(8).index.tolist()
        box_df = df[df["primary_category"].isin(top8)]
        fig_box = px.box(
            box_df, x="primary_category", y="BUSINESS_AVG_STARS",
            title="Star Rating Distribution by Top 8 Categories",
            labels={"primary_category": "Category",
                    "BUSINESS_AVG_STARS": "Avg Stars"},
            color="primary_category",
        )
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EXPLORATION
#  Interactive filters (state, city, category, rating range, open/closed)
#  and a dataset preview with summary statistics.
# ════════════════════════════════════════════════════════════════════════════════
with tab_explore:

    st.markdown('<p class="section-title">🔎 Filter & Explore Restaurants</p>',
                unsafe_allow_html=True)

    # ── Filter controls ────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        # State picker — "All" means no state filter
        state_options = ["All"] + sorted(df["STATE"].dropna().unique().tolist())
        selected_state = st.selectbox("State / Province", state_options, index=0)

    with f2:
        # City picker — dynamically updates based on selected state
        if selected_state == "All":
            city_pool = df["CITY"].dropna().unique().tolist()
        else:
            city_pool = df[df["STATE"] == selected_state]["CITY"].dropna().unique().tolist()
        city_options = ["All"] + sorted(city_pool)
        selected_city = st.selectbox("City", city_options, index=0)

    with f3:
        # Category multi-select
        cat_options = sorted(df["primary_category"].dropna().unique().tolist())
        selected_cats = st.multiselect("Category", cat_options, default=[])

    f4, f5, f6 = st.columns(3)

    with f4:
        # Star rating range slider
        rating_range = st.slider(
            "Minimum Average Stars", 1.0, 5.0, 1.0, step=0.5
        )

    with f5:
        # Open/Closed filter
        open_filter = st.radio(
            "Business Status", ["All", "Open Only", "Closed Only"],
            horizontal=True
        )

    with f6:
        # Review count threshold
        min_reviews = st.number_input(
            "Min Review Count", min_value=0, max_value=5000, value=0, step=10
        )

    # ── Apply all filters to the DataFrame ────────────────────────────────────
    filtered = df.copy()

    if selected_state != "All":
        filtered = filtered[filtered["STATE"] == selected_state]
    if selected_city != "All":
        filtered = filtered[filtered["CITY"] == selected_city]
    if selected_cats:
        filtered = filtered[filtered["primary_category"].isin(selected_cats)]

    filtered = filtered[filtered["BUSINESS_AVG_STARS"] >= rating_range]
    filtered = filtered[filtered["BUSINESS_REVIEW_COUNT"] >= min_reviews]

    if open_filter == "Open Only":
        filtered = filtered[filtered["IS_OPEN"] == 1]
    elif open_filter == "Closed Only":
        filtered = filtered[filtered["IS_OPEN"] == 0]

    # ── Results count banner ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #E8450A44;border-radius:10px;
                padding:0.8rem 1.2rem;margin:0.5rem 0 1rem">
        <b style="color:#E8450A;font-size:1.2rem">{len(filtered):,}</b>
        <span style="color:#aaa"> restaurants match your filters
        (out of {len(df):,} total)</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts for filtered data ───────────────────────────────────────────────
    if len(filtered) > 0:
        ch1, ch2 = st.columns(2)

        with ch1:
            # Category breakdown for filtered results
            cat_bar = filtered["primary_category"].value_counts().head(12).reset_index()
            cat_bar.columns = ["Category", "Count"]
            fig_cat_filter = px.bar(
                cat_bar, x="Category", y="Count",
                title="Category Breakdown (Filtered)",
                color="Count",
                color_continuous_scale=["#2a2a3e", "#E8450A"],
            )
            fig_cat_filter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-30,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat_filter, use_container_width=True)

        with ch2:
            # Rating histogram for filtered results
            fig_hist_filter = px.histogram(
                filtered, x="BUSINESS_AVG_STARS",
                nbins=10,
                title="Rating Distribution (Filtered)",
                color_discrete_sequence=["#E8450A"],
            )
            fig_hist_filter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist_filter, use_container_width=True)

        # Top restaurants by review count in filtered set
        st.markdown('<p class="section-title">Top Restaurants by Review Count</p>',
                    unsafe_allow_html=True)

        top_reviewed = (
            filtered[["BUSINESS_NAME", "CITY", "STATE",
                       "primary_category", "BUSINESS_AVG_STARS",
                       "BUSINESS_REVIEW_COUNT", "IS_OPEN_LABEL"]]
            .sort_values("BUSINESS_REVIEW_COUNT", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        top_reviewed.index = top_reviewed.index + 1  # 1-based rank

        # Style the DataFrame for display
        st.dataframe(
            top_reviewed.rename(columns={
                "BUSINESS_NAME":       "Name",
                "CITY":                "City",
                "STATE":               "State",
                "primary_category":    "Category",
                "BUSINESS_AVG_STARS":  "Avg ⭐",
                "BUSINESS_REVIEW_COUNT": "Reviews",
                "IS_OPEN_LABEL":       "Status",
            }),
            use_container_width=True,
            height=480,
        )

        # ── Rating vs Engagement scatter (filtered) ────────────────────────────
        st.markdown('<p class="section-title">Rating vs Engagement (Filtered)</p>',
                    unsafe_allow_html=True)

        fig_engage = px.scatter(
            filtered.sample(min(1500, len(filtered)), random_state=42),
            x="TOTAL_REVIEW_USEFUL_COUNT",
            y="BUSINESS_AVG_STARS",
            color="primary_category",
            size="BUSINESS_REVIEW_COUNT",
            hover_name="BUSINESS_NAME",
            title="Useful Votes vs Rating — sized by Review Count",
            labels={
                "TOTAL_REVIEW_USEFUL_COUNT": "Total Useful Votes",
                "BUSINESS_AVG_STARS":         "Avg Stars",
            },
            opacity=0.65,
        )
        fig_engage.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_engage, use_container_width=True)

    else:
        st.warning("No restaurants match the selected filters. Try relaxing some criteria.")


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — RECOMMENDATION
#  The main prediction interface.
#
#  Two user modes:
#    • Existing user  → User ID input → Hybrid SVD + KNN (+ optional Pinecone)
#    • New user       → Preference text input only → Pinecone semantic search
#
#  The Pinecone connection is lazy-loaded on first use.
# ════════════════════════════════════════════════════════════════════════════════
with tab_recommend:

    st.markdown('<p class="section-title">🤖 Get Personalised Restaurant Recommendations</p>',
                unsafe_allow_html=True)

    # ── Explanation of the three models ───────────────────────────────────────
    with st.expander("ℹ️  How the recommendation engine works", expanded=False):
        st.markdown("""
        | Model | Type | When used |
        |---|---|---|
        | **SVD** (Singular Value Decomposition) | Collaborative Filtering | Existing users |
        | **KNN** (K-Nearest Neighbours) | Collaborative Filtering | Existing users |
        | **Pinecone** (sentence-transformers embeddings) | Semantic / Content-based | New users & preference boost |

        **Hybrid logic for existing users:**
        - SVD + KNN ratings are predicted, normalised and blended (configurable weights).
        - If preference text is provided, a Pinecone similarity score is also computed and mixed in (70 % CF + 30 % semantic).

        **New users / unknown IDs:** Only Pinecone semantic search is used — no historical ratings needed.
        """)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("rec_form"):

        r1, r2 = st.columns([1, 1])

        with r1:
            user_id_input = st.text_input(
                "User ID (leave blank if you're a new user)",
                placeholder="e.g. qLBBMSKl8GwHD1tBMIpWeg",
                help="Must be a User ID that exists in the training dataset. "
                     "Leave blank to use preference-text only (new user mode)."
            )
            preference_text = st.text_area(
                "Preference / Vibe Description",
                placeholder="e.g. cozy Italian restaurant with great wine and pasta",
                height=100,
                help="Used as Pinecone semantic query. Required for new users; "
                     "optional boost for existing users."
            )

        with r2:
            # Location filters — optional
            loc1, loc2 = st.columns(2)
            with loc1:
                loc_state = st.selectbox(
                    "State Filter",
                    [""] + sorted(df["STATE"].dropna().unique().tolist()),
                    index=0,
                )
            with loc2:
                if loc_state:
                    city_pool_rec = df[df["STATE"] == loc_state]["CITY"].dropna().unique().tolist()
                else:
                    city_pool_rec = df["CITY"].dropna().unique().tolist()
                loc_city = st.selectbox(
                    "City Filter",
                    [""] + sorted(city_pool_rec),
                    index=0,
                )

            top_k = st.slider(
                "Number of Recommendations", min_value=1, max_value=15, value=5
            )

            # Hybrid weight sliders (only matter for existing users)
            st.markdown("**SVD / KNN blend** *(existing users only)*")
            svd_w = st.slider("SVD Weight", 0.0, 1.0, 0.6, 0.05)
            knn_w = round(1.0 - svd_w, 2)
            st.caption(f"KNN weight: **{knn_w}** (auto-computed as 1 − SVD)")

        submitted = st.form_submit_button("🍽️  Get Recommendations", type="primary")

    # ── Process the recommendation request ────────────────────────────────────
    if submitted:

        uid   = user_id_input.strip() or None
        ptext = preference_text.strip() or None
        city  = loc_city  or None
        state = loc_state or None

        # Guard: new user must provide preference text
        if not uid and not ptext:
            st.error("Please enter either a User ID or a preference description (or both).")
            st.stop()

        with st.spinner("Computing recommendations…"):
            results, strategy = master_recommend(
                user_id        = uid,
                preference_text= ptext,
                city           = city,
                state          = state,
                top_k          = top_k,
                svd_model      = svd_model,
                knn_model      = knn_model,
                user_encoder   = user_enc,
                biz_encoder    = biz_enc,
                train_df       = train_df,
                business_meta  = biz_meta,
                svd_weight     = svd_w,
                knn_weight     = knn_w,
            )

        # ── Results header ─────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #E8450A44;border-radius:10px;
                    padding:0.8rem 1.2rem;margin:1rem 0">
            <b style="color:#E8450A">Strategy:</b>
            <span style="color:#fff"> {strategy}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b style="color:#E8450A">Results:</b>
            <span style="color:#fff"> {len(results) if not results.empty else 0} restaurant(s)</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Render result cards ────────────────────────────────────────────────
        if results.empty:
            st.warning(
                "No recommendations found. Try a different location, loosen filters, "
                "or provide more descriptive preference text."
            )
        else:
            has_desc = "description" in results.columns

            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_rec_card(rank, row, has_description=has_desc)

            # ── Score comparison chart (if multiple results) ──────────────────
            if len(results) > 1 and "final_score" in results.columns:
                st.markdown('<p class="section-title">Score Comparison</p>',
                            unsafe_allow_html=True)

                name_col = "business_name" if "business_name" in results.columns else "BUSINESS_NAME"
                chart_df = results[[name_col, "final_score"]].copy()
                chart_df.columns = ["Restaurant", "Score"]

                # If hybrid scores exist, show SVD and KNN individually
                extra_traces = []
                if "svd_score" in results.columns:
                    extra_traces.append(("SVD Score", results["svd_score"].tolist(), "#4ecdc4"))
                if "knn_score" in results.columns:
                    extra_traces.append(("KNN Score", results["knn_score"].tolist(), "#ffe66d"))
                if "pine_score" in results.columns:
                    extra_traces.append(("Pinecone Score", results["pine_score"].tolist(), "#a29bfe"))

                fig_score = go.Figure()

                # Final blended score bar
                fig_score.add_trace(go.Bar(
                    name="Final Score",
                    x=chart_df["Restaurant"],
                    y=chart_df["Score"],
                    marker_color="#E8450A",
                ))

                # Individual model score lines on top
                for trace_name, vals, colour in extra_traces:
                    fig_score.add_trace(go.Scatter(
                        name=trace_name,
                        x=chart_df["Restaurant"],
                        y=vals,
                        mode="lines+markers",
                        marker_color=colour,
                        line=dict(width=2),
                    ))

                fig_score.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    barmode="overlay",
                    xaxis_tickangle=-20,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=380,
                )
                st.plotly_chart(fig_score, use_container_width=True)

    # ── Quick demo hint when form hasn't been submitted yet ───────────────────
    else:
        # SVD-only is enough for the hybrid path; KNN is a bonus
        models_ready = svd_model is not None

        if not models_ready:
            # Warn the user that CF models are unavailable but Pinecone still works
            st.warning(
                "⚠️ **SVD model is not loaded** (scikit-surprise is not installed).  \n\n"
                "The **Recommendation tab still works** via Pinecone semantic search — "
                "just leave the User ID blank and describe what you're looking for.  \n\n"
                "To enable full hybrid recommendations:  \n"
                "```\nconda install -c conda-forge scikit-surprise\n```"
            )
        else:
            st.info(
                "💡 **Quick start:** Leave the User ID blank and type a preference like "
                "*'outdoor brunch spot with great coffee'* to try the semantic search. "
                "For personalised results, enter an existing User ID from the dataset."
            )

        # Show sample User IDs only when the CF models are loaded
        if not train_df.empty and models_ready:
            sample_ids = train_df["user_id"].drop_duplicates().head(5).tolist()
            st.markdown("**Sample User IDs** (copy one to try the hybrid model):")
            st.code("\n".join(sample_ids), language="text")
        elif models_ready is False:
            st.info(
                "💡 **Try it now:** Type a preference below, e.g.  \n"
                "*'spicy Mexican food with great margaritas'*  \n"
                "*'cozy Italian with outdoor seating'*  \n"
                "*'best sushi bar in the city'*"
            )
