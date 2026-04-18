import streamlit as st
import plotly.express as px


def render_overview(df):

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
        top_states = df["STATE"].value_counts().head(10).reset_index()
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
        top_cats = df["primary_category"].value_counts().head(10).reset_index()
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
