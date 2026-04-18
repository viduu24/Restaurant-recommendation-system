import streamlit as st
import plotly.graph_objects as go


def star_html(rating):
    full  = int(round(rating))
    stars = "★" * full + "☆" * (5 - full)
    return f'<span style="color:#FFD700;font-size:1.1rem">{stars}</span> {rating:.1f}'


def render_rec_card(rank, row, has_description=False):
    name  = row.get("business_name", row.get("BUSINESS_NAME", "N/A"))
    city  = row.get("city",  row.get("CITY",  ""))
    state = row.get("state", row.get("STATE", ""))
    cat   = row.get("primary_category", row.get("categories", ""))
    stars = float(row.get("business_avg_stars", row.get("avg_stars", 0)))
    score = row.get("final_score", None)
    desc  = row.get("description", "") if has_description else ""

    score_badge = (
        f'<span class="rec-badge">Score: {score:.3f}</span>' if score is not None else ""
    )
    desc_html = (
        f'<p style="color:#ccc;font-size:0.88rem;margin-top:0.5rem">{desc}</p>'
        if desc else ""
    )

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


def render_recommendation(df, svd_model, knn_model, user_enc, biz_enc,
                           train_df, biz_meta, master_recommend):

    st.markdown('<p class="section-title">🤖 Get Personalised Restaurant Recommendations</p>',
                unsafe_allow_html=True)

    with st.expander("ℹ️  How the recommendation engine works", expanded=False):
        st.markdown("""
        | Model | Type | When used |
        |---|---|---|
        | **SVD** | Collaborative Filtering | Existing users |
        | **KNN** | Collaborative Filtering | Existing users |
        | **Pinecone** | Semantic / Content-based | New users & preference boost |

        **Hybrid logic for existing users:**
        - SVD + KNN ratings are predicted, normalised and blended (configurable weights).
        - If preference text is provided, a Pinecone similarity score is also mixed in (70% CF + 30% semantic).

        **New users / unknown IDs:** Only Pinecone semantic search is used.
        """)

    with st.form("rec_form"):
        r1, r2 = st.columns([1, 1])

        with r1:
            user_id_input = st.text_input(
                "User ID (leave blank if you're a new user)",
                placeholder="e.g. qLBBMSKl8GwHD1tBMIpWeg",
            )
            preference_text = st.text_area(
                "Preference / Vibe Description",
                placeholder="e.g. cozy Italian restaurant with great wine and pasta",
                height=100,
            )

        with r2:
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

            top_k = st.slider("Number of Recommendations", min_value=1, max_value=15, value=5)

            st.markdown("**SVD / KNN blend** *(existing users only)*")
            svd_w = st.slider("SVD Weight", 0.0, 1.0, 0.6, 0.05)
            knn_w = round(1.0 - svd_w, 2)
            st.caption(f"KNN weight: **{knn_w}** (auto-computed as 1 − SVD)")

        submitted = st.form_submit_button("🍽️  Get Recommendations", type="primary")

    if submitted:
        uid   = user_id_input.strip() or None
        ptext = preference_text.strip() or None
        city  = loc_city  or None
        state = loc_state or None

        if not uid and not ptext:
            st.error("Please enter either a User ID or a preference description (or both).")
            st.stop()

        with st.spinner("Computing recommendations…"):
            results, strategy = master_recommend(
                user_id         = uid,
                preference_text = ptext,
                city            = city,
                state           = state,
                top_k           = top_k,
                svd_model       = svd_model,
                knn_model       = knn_model,
                user_encoder    = user_enc,
                biz_encoder     = biz_enc,
                train_df        = train_df,
                business_meta   = biz_meta,
                svd_weight      = svd_w,
                knn_weight      = knn_w,
            )

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

        if results.empty:
            st.warning(
                "No recommendations found. Try a different location, loosen filters, "
                "or provide more descriptive preference text."
            )
        else:
            has_desc = "description" in results.columns
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                render_rec_card(rank, row, has_description=has_desc)

            if len(results) > 1 and "final_score" in results.columns:
                st.markdown('<p class="section-title">Score Comparison</p>',
                            unsafe_allow_html=True)

                name_col = "business_name" if "business_name" in results.columns else "BUSINESS_NAME"
                chart_df = results[[name_col, "final_score"]].copy()
                chart_df.columns = ["Restaurant", "Score"]

                extra_traces = []
                if "svd_score"  in results.columns:
                    extra_traces.append(("SVD Score",      results["svd_score"].tolist(),  "#4ecdc4"))
                if "knn_score"  in results.columns:
                    extra_traces.append(("KNN Score",      results["knn_score"].tolist(),  "#ffe66d"))
                if "pine_score" in results.columns:
                    extra_traces.append(("Pinecone Score", results["pine_score"].tolist(), "#a29bfe"))

                fig_score = go.Figure()
                fig_score.add_trace(go.Bar(
                    name="Final Score",
                    x=chart_df["Restaurant"],
                    y=chart_df["Score"],
                    marker_color="#E8450A",
                ))
                for trace_name, vals, colour in extra_traces:
                    fig_score.add_trace(go.Scatter(
                        name=trace_name, x=chart_df["Restaurant"], y=vals,
                        mode="lines+markers", marker_color=colour,
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

    else:
        models_ready = svd_model is not None

        if not models_ready:
            st.warning(
                "⚠️ **SVD model is not loaded.**  \n\n"
                "The Recommendation tab still works via Pinecone — leave the User ID "
                "blank and describe what you're looking for.  \n\n"
                "To enable full hybrid mode:  \n"
                "```\nconda install -c conda-forge scikit-surprise\n```"
            )
        else:
            st.info(
                "💡 Leave the User ID blank and type a preference like "
                "*'outdoor brunch spot with great coffee'* to try semantic search. "
                "For personalised results, enter an existing User ID."
            )

        if not train_df.empty and models_ready:
            sample_ids = train_df["user_id"].drop_duplicates().head(5).tolist()
            st.markdown("**Sample User IDs** (copy one to try the hybrid model):")
            st.code("\n".join(sample_ids), language="text")
