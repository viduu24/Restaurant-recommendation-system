import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Business Recommendation Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# Data Path
# Update this path only if your folder location changes
# ---------------------------------------------------
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"

# ---------------------------------------------------
# Load Data
# cache_data helps Streamlit avoid reloading large files
# every time the app refreshes
# ---------------------------------------------------
@st.cache_data
def load_data():
    ratings_train = pd.read_csv(DATA_PATH / "ratings_train.csv")
    ratings_test = pd.read_csv(DATA_PATH / "ratings_test.csv")
    business_meta = pd.read_csv(DATA_PATH / "business_meta.csv")
    return ratings_train, ratings_test, business_meta

# ---------------------------------------------------
# Merge ratings with business metadata
# This allows us to analyze ratings with business name,
# location, categories, and average stars
# ---------------------------------------------------
@st.cache_data
def prepare_data():
    ratings_train, ratings_test, business_meta = load_data()

    train_merged = ratings_train.merge(business_meta, on="business_id", how="left")
    test_merged = ratings_test.merge(business_meta, on="business_id", how="left")

    return ratings_train, ratings_test, business_meta, train_merged, test_merged

# ---------------------------------------------------
# Prepare category data for category-based analysis
# The categories column has multiple values in one cell,
# so we split and explode it into separate rows
# ---------------------------------------------------
@st.cache_data
def prepare_category_data(business_meta):
    category_df = business_meta[["business_id", "categories"]].dropna().copy()
    category_df["categories"] = category_df["categories"].astype(str).str.split(", ")
    category_df = category_df.explode("categories")
    category_df = category_df.dropna(subset=["categories"])
    return category_df

# ---------------------------------------------------
# Prepare recommendation table
# We calculate:
# 1. average rating from users
# 2. total number of ratings
# 3. recommendation score
# ---------------------------------------------------
@st.cache_data
def prepare_recommendation_data(train_merged):
    recommendation_df = (
        train_merged.groupby(
            ["business_id", "business_name", "city", "state", "business_avg_stars", "categories"],
            dropna=False
        )
        .agg(
            user_avg_rating=("target_rating", "mean"),
            rating_count=("target_rating", "count")
        )
        .reset_index()
    )

    recommendation_df["user_avg_rating"] = recommendation_df["user_avg_rating"].round(2)
    recommendation_df["business_avg_stars"] = pd.to_numeric(
        recommendation_df["business_avg_stars"], errors="coerce"
    )

    # Simple recommendation score
    recommendation_df["recommendation_score"] = (
        recommendation_df["user_avg_rating"] * 0.7 +
        recommendation_df["business_avg_stars"].fillna(0) * 0.3
    ).round(2)

    return recommendation_df

# ---------------------------------------------------
# Load all datasets
# ---------------------------------------------------
ratings_train, ratings_test, business_meta, train_merged, test_merged = prepare_data()
category_df = prepare_category_data(business_meta)
recommendation_df = prepare_recommendation_data(train_merged)

# ---------------------------------------------------
# Sidebar Navigation
# The user chooses which page to open
# ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Exploration", "Recommendation"]
)

# ---------------------------------------------------
# Sidebar Filters
# These filters are mainly used on Exploration page
# ---------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Common Filters")

state_options = sorted(train_merged["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Select State", state_options)

filtered_df = train_merged.copy()

if selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]

city_options = sorted(filtered_df["city"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("Select City", city_options)

if selected_cities:
    filtered_df = filtered_df[filtered_df["city"].isin(selected_cities)]

rating_options = sorted(filtered_df["target_rating"].dropna().unique().tolist())
selected_ratings = st.sidebar.multiselect("Select Rating", rating_options)

if selected_ratings:
    filtered_df = filtered_df[filtered_df["target_rating"].isin(selected_ratings)]

# ===================================================
# PAGE 1: OVERVIEW
# ===================================================
if page == "Overview":
    st.title("Overview")
    st.markdown(
        """
        This page gives a high-level summary of the datasets.
        It helps us understand the size of the data, rating behavior,
        and overall business information before deeper exploration.
        """
    )

    # -------------------------------
    # KPI cards
    # -------------------------------
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Train Ratings", f"{len(ratings_train):,}")
    col2.metric("Test Ratings", f"{len(ratings_test):,}")
    col3.metric("Unique Users", f"{ratings_train['user_id'].nunique():,}")
    col4.metric("Unique Businesses", f"{business_meta['business_id'].nunique():,}")
    col5.metric("Cities", f"{business_meta['city'].nunique():,}")
    col6.metric("States", f"{business_meta['state'].nunique():,}")

    st.markdown("---")

    # -------------------------------
    # Dataset shapes
    # -------------------------------
    st.subheader("Dataset Shapes")
    st.write("Ratings Train Shape:", ratings_train.shape)
    st.write("Ratings Test Shape:", ratings_test.shape)
    st.write("Business Meta Shape:", business_meta.shape)

    st.markdown("---")

    # -------------------------------
    # Missing values summary
    # -------------------------------
    st.subheader("Missing Values Summary")
    missing_df = pd.DataFrame({
        "ratings_train": ratings_train.isnull().sum(),
        "ratings_test": ratings_test.isnull().sum(),
        "business_meta": business_meta.isnull().sum()
    }).fillna("")
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # Ratings distribution
    # -------------------------------
    st.subheader("Training Ratings Distribution")
    st.markdown(
        """
        This chart shows how ratings are distributed in the training dataset.
        It helps identify whether users give more high ratings or low ratings.
        """
    )

    fig_rating_dist = px.histogram(
        ratings_train,
        x="target_rating",
        nbins=5,
        title="Distribution of Training Ratings"
    )
    st.plotly_chart(fig_rating_dist, use_container_width=True)

    # -------------------------------
    # Business average stars distribution
    # -------------------------------
    st.subheader("Business Average Stars Distribution")
    st.markdown(
        """
        This chart shows the spread of average business ratings from the metadata file.
        It helps us understand the general quality level of businesses in the dataset.
        """
    )

    fig_business_stars = px.histogram(
        business_meta,
        x="business_avg_stars",
        nbins=20,
        title="Distribution of Business Average Stars"
    )
    st.plotly_chart(fig_business_stars, use_container_width=True)

    # -------------------------------
    # Train vs test quick comparison
    # -------------------------------
    st.subheader("Train vs Test Summary")
    comparison_df = pd.DataFrame({
        "Metric": ["Rows", "Unique Users", "Unique Businesses"],
        "Train": [
            len(ratings_train),
            ratings_train["user_id"].nunique(),
            ratings_train["business_id"].nunique()
        ],
        "Test": [
            len(ratings_test),
            ratings_test["user_id"].nunique(),
            ratings_test["business_id"].nunique()
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)

# ===================================================
# PAGE 2: EXPLORATION
# ===================================================
elif page == "Exploration":
    st.title("Exploration")
    st.markdown(
        """
        This page is for interactive data analysis.
        You can use the sidebar filters to explore how ratings vary by business,
        city, state, and category.
        """
    )

    st.info(
        "Use the sidebar filters to narrow the data by state, city, and rating."
    )

    # -------------------------------
    # Show filtered data size
    # -------------------------------
    st.subheader("Filtered Data Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Rows", f"{len(filtered_df):,}")
    col2.metric("Filtered Businesses", f"{filtered_df['business_id'].nunique():,}")
    col3.metric("Filtered Users", f"{filtered_df['user_id'].nunique():,}")

    st.markdown("---")

    # -------------------------------
    # Top most-rated businesses
    # -------------------------------
    st.subheader("Top 10 Most Rated Businesses")
    st.markdown(
        """
        This chart shows which businesses received the largest number of ratings.
        It helps identify the most popular businesses in the filtered data.
        """
    )

    top_businesses = (
        filtered_df.groupby("business_name")
        .size()
        .reset_index(name="rating_count")
        .sort_values("rating_count", ascending=False)
        .head(10)
    )

    fig_top_businesses = px.bar(
        top_businesses,
        x="rating_count",
        y="business_name",
        orientation="h",
        title="Top 10 Most Rated Businesses"
    )
    st.plotly_chart(fig_top_businesses, use_container_width=True)

    # -------------------------------
    # Ratings by city
    # -------------------------------
    st.subheader("Top Cities by Number of Ratings")
    st.markdown(
        """
        This chart shows which cities contribute the highest number of ratings.
        It helps reveal geographical concentration in the dataset.
        """
    )

    city_ratings = (
        filtered_df.groupby("city")
        .size()
        .reset_index(name="rating_count")
        .sort_values("rating_count", ascending=False)
        .head(15)
    )

    fig_city = px.bar(
        city_ratings,
        x="city",
        y="rating_count",
        title="Top Cities by Ratings"
    )
    st.plotly_chart(fig_city, use_container_width=True)

    # -------------------------------
    # Ratings by state
    # -------------------------------
    st.subheader("Ratings by State")
    st.markdown(
        """
        This chart compares rating counts across states.
        It helps us see which states have the strongest activity.
        """
    )

    state_ratings = (
        filtered_df.groupby("state")
        .size()
        .reset_index(name="rating_count")
        .sort_values("rating_count", ascending=False)
    )

    fig_state = px.bar(
        state_ratings,
        x="state",
        y="rating_count",
        title="Ratings by State"
    )
    st.plotly_chart(fig_state, use_container_width=True)

    # -------------------------------
    # Top categories
    # -------------------------------
    st.subheader("Top Business Categories")
    st.markdown(
        """
        Since one business can belong to multiple categories,
        we split the category column and count category frequency.
        This chart shows the most common business categories.
        """
    )

    top_categories = (
        category_df["categories"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_categories.columns = ["category", "count"]

    fig_categories = px.bar(
        top_categories,
        x="count",
        y="category",
        orientation="h",
        title="Top 15 Categories"
    )
    st.plotly_chart(fig_categories, use_container_width=True)

    # -------------------------------
    # Business search
    # -------------------------------
    st.subheader("Business Search")
    st.markdown(
        """
        Select a business to inspect its details such as city, state,
        average stars, categories, and number of ratings.
        """
    )

    business_names = sorted(filtered_df["business_name"].dropna().unique().tolist())

    if business_names:
        selected_business = st.selectbox("Select a Business", business_names)

        business_details = filtered_df[filtered_df["business_name"] == selected_business]

        if not business_details.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.write("**Business Name:**", selected_business)
                st.write("**City:**", business_details["city"].iloc[0])
                st.write("**State:**", business_details["state"].iloc[0])
                st.write("**Business Avg Stars:**", business_details["business_avg_stars"].iloc[0])

            with c2:
                st.write("**Categories:**", business_details["categories"].iloc[0])
                st.write("**Number of User Ratings:**", len(business_details))
                st.write("**Average User Rating:**", round(business_details["target_rating"].mean(), 2))

            st.dataframe(
                business_details[[
                    "user_id", "business_id", "business_name",
                    "city", "state", "target_rating",
                    "business_avg_stars", "categories"
                ]].head(100),
                use_container_width=True
            )
    else:
        st.warning("No businesses available for the current filter selection.")

# ===================================================
# PAGE 3: RECOMMENDATION
# ===================================================
elif page == "Recommendation":
    st.title("Recommendation")
    st.markdown(
        """
        This page provides a simple recommendation system based on business popularity
        and rating quality.

        The recommendation logic uses:
        - average user rating from training data
        - business average stars from metadata
        - minimum number of ratings for reliability
        """
    )

    # -------------------------------
    # Recommendation filters
    # -------------------------------
    st.subheader("Recommendation Filters")

    rec_df = recommendation_df.copy()

    rec_states = sorted(rec_df["state"].dropna().unique().tolist())
    selected_rec_states = st.multiselect("Filter by State", rec_states)

    if selected_rec_states:
        rec_df = rec_df[rec_df["state"].isin(selected_rec_states)]

    rec_cities = sorted(rec_df["city"].dropna().unique().tolist())
    selected_rec_cities = st.multiselect("Filter by City", rec_cities)

    if selected_rec_cities:
        rec_df = rec_df[rec_df["city"].isin(selected_rec_cities)]

    min_ratings = st.slider("Minimum Number of Ratings", 1, 200, 20)
    rec_df = rec_df[rec_df["rating_count"] >= min_ratings]

    category_keyword = st.text_input("Filter by Category Keyword")
    if category_keyword:
        rec_df = rec_df[
            rec_df["categories"].astype(str).str.contains(category_keyword, case=False, na=False)
        ]

    st.markdown("---")

    # -------------------------------
    # Recommendation explanation
    # -------------------------------
    st.subheader("How Recommendation Score is Calculated")
    st.markdown(
        """
        The recommendation score is calculated using:

        **Recommendation Score = 0.7 × User Average Rating + 0.3 × Business Average Stars**

        This gives more weight to actual user ratings while still considering
        the overall business reputation from metadata.
        """
    )

    # -------------------------------
    # Top recommendations table
    # -------------------------------
    st.subheader("Top Recommended Businesses")
    st.markdown(
        """
        This table shows the highest-ranked businesses after applying the selected filters.
        Businesses with very low rating counts are filtered out to improve reliability.
        """
    )

    top_recommendations = rec_df.sort_values(
        "recommendation_score", ascending=False
    ).head(20)

    if not top_recommendations.empty:
        st.dataframe(
            top_recommendations[[
                "business_name",
                "city",
                "state",
                "user_avg_rating",
                "business_avg_stars",
                "rating_count",
                "recommendation_score",
                "categories"
            ]],
            use_container_width=True
        )
    else:
        st.warning("No recommendation results found for the selected filters.")

    # -------------------------------
    # Top recommendation chart
    # -------------------------------
    st.subheader("Top 10 Recommended Businesses")
    st.markdown(
        """
        This chart visualizes the top recommended businesses based on the
        calculated recommendation score.
        """
    )

    top_chart = rec_df.sort_values(
        "recommendation_score", ascending=False
    ).head(10)

    if not top_chart.empty:
        fig_recommend = px.bar(
            top_chart,
            x="recommendation_score",
            y="business_name",
            orientation="h",
            hover_data=["city", "state", "user_avg_rating", "rating_count"],
            title="Top 10 Recommended Businesses"
        )
        st.plotly_chart(fig_recommend, use_container_width=True)
    else:
        st.warning("No chart data available for the current filters.")

    # -------------------------------
    # Popularity vs rating analysis
    # -------------------------------
    st.subheader("Popularity vs User Average Rating")
    st.markdown(
        """
        This scatter plot compares how popular a business is
        (number of ratings) versus how well users rate it.
        This helps identify businesses that are both reliable and highly rated.
        """
    )

    scatter_df = rec_df.dropna(subset=["rating_count", "user_avg_rating", "business_name"]).copy()

    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df.head(1000),
            x="rating_count",
            y="user_avg_rating",
            hover_name="business_name",
            title="Popularity vs User Average Rating"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No scatter plot data available for the current filters.")