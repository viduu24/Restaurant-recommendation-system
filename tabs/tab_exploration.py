import streamlit as st
import plotly.express as px


def render_exploration(df):

    st.markdown('<p class="section-title">🔎 Filter & Explore Restaurants</p>',
                unsafe_allow_html=True)

    # ── Filter controls ────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        state_options = ["All"] + sorted(df["STATE"].dropna().unique().tolist())
        selected_state = st.selectbox("State / Province", state_options, index=0)

    with f2:
        if selected_state == "All":
            city_pool = df["CITY"].dropna().unique().tolist()
        else:
            city_pool = df[df["STATE"] == selected_state]["CITY"].dropna().unique().tolist()
        city_options = ["All"] + sorted(city_pool)
        selected_city = st.selectbox("City", city_options, index=0)

    with f3:
        cat_options = sorted(df["primary_category"].dropna().unique().tolist())
        selected_cats = st.multiselect("Category", cat_options, default=[])

    f4, f5, f6 = st.columns(3)

    with f4:
        rating_range = st.slider("Minimum Average Stars", 1.0, 5.0, 1.0, step=0.5)

    with f5:
        open_filter = st.radio(
            "Business Status", ["All", "Open Only", "Closed Only"],
            horizontal=True
        )

    with f6:
        min_reviews = st.number_input(
            "Min Review Count", min_value=0, max_value=5000, value=0, step=10
        )

    # ── Apply filters ──────────────────────────────────────────────────────────
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

    if len(filtered) > 0:
        ch1, ch2 = st.columns(2)

        with ch1:
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
        top_reviewed.index = top_reviewed.index + 1

        st.dataframe(
            top_reviewed.rename(columns={
                "BUSINESS_NAME":         "Name",
                "CITY":                  "City",
                "STATE":                 "State",
                "primary_category":      "Category",
                "BUSINESS_AVG_STARS":    "Avg ⭐",
                "BUSINESS_REVIEW_COUNT": "Reviews",
                "IS_OPEN_LABEL":         "Status",
            }),
            use_container_width=True,
            height=480,
        )

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
                "BUSINESS_AVG_STARS":        "Avg Stars",
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
