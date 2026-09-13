import streamlit as st
import pandas as pd
from pathlib import Path
from trend_pipeline import topic_names
from trend_pipeline import analyse_posts

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Social Media Trend Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 15px;
        border-radius: 12px;
    }

    h1 {
        margin-bottom: 0.3rem;
    }

    h2, h3 {
        margin-top: 1.2rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    final_ranking = pd.read_csv(
        DATA_DIR / "final_ranking.csv"
    )

    trend_intelligence = pd.read_csv(
        DATA_DIR / "trend_intelligence.csv"
    )

    topic_daily = pd.read_csv(
        DATA_DIR / "topic_daily.csv"
    )

    posts = pd.read_csv(
        DATA_DIR / "posts.csv"
    )

    # Convert date columns
    if "date" in topic_daily.columns:
        topic_daily["date"] = pd.to_datetime(
            topic_daily["date"],
            errors="coerce"
        )

    if "date" in posts.columns:
        posts["date"] = pd.to_datetime(
            posts["date"],
            errors="coerce"
        )

    # Make score numeric
    if "score" in posts.columns:
        posts["score"] = pd.to_numeric(
            posts["score"],
            errors="coerce"
        )

    return (
        final_ranking,
        trend_intelligence,
        topic_daily,
        posts
    )


try:

    (
        final_ranking,
        trend_intelligence,
        topic_daily,
        posts
    ) = load_data()

except FileNotFoundError:

    st.error(
        "Processed CSV files were not found.\n\n"
        "Make sure these files exist inside "
        "`data/processed/`."
    )

    st.stop()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def trend_icon(trend):

    icons = {
        "Rising": "📈",
        "Stable": "➡️",
        "Falling": "📉"
    }

    return icons.get(
        trend,
        "📊"
    )


def signal_icon(signal):

    icons = {
        "Emerging Trend": "🚀",
        "Possible Rebound": "🔄",
        "Stabilizing": "⚖️",
        "Steady": "➡️",
        "Continuing Rise": "🔥",
        "Cooling Down": "❄️",
        "Possible Decline": "⚠️",
        "Continuing Decline": "📉"
    }

    return icons.get(
        signal,
        "📊"
    )


signal_explanations = {

    "Emerging Trend":
        "Activity has upward potential and the model predicts a rise.",

    "Possible Rebound":
        "The topic is currently weak or falling but may recover.",

    "Stabilizing":
        "The topic is currently strong but may begin to level off.",

    "Steady":
        "The topic is expected to remain relatively stable.",

    "Continuing Rise":
        "The topic is already rising and may continue rising.",

    "Cooling Down":
        "The topic is currently strong but may begin losing momentum.",

    "Possible Decline":
        "The topic may begin losing momentum.",

    "Continuing Decline":
        "The topic is already falling and may continue declining."
}


def safe_score(value):

    try:

        value = float(value)

        value = max(
            0,
            min(
                value,
                100
            )
        )

        return value

    except:

        return 0.0


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Trend Analyser"
)

st.sidebar.caption(
    "Explore AI discussions, emerging trends "
    "and Reddit engagement."
)
st.sidebar.markdown("### Data Source")

data_source = st.sidebar.radio(
    "Choose analysis source",
    [
        "Historical Dataset",
        "New Data Analysis"
    ],
    label_visibility="collapsed"
)
if data_source == "Historical Dataset":

    # YOUR EXISTING DASHBOARD CODE

  page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔎 Topic Explorer"
    ]
  )

  st.sidebar.divider()

  st.sidebar.caption(
    "Source: AI-related Reddit communities")


# ============================================================
# 7. OVERVIEW PAGE
# ============================================================

  if page == "🏠 Overview":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "📈 AI-Powered Social Media Trend Analyser"
    )

    st.write(
        """
        Discover which AI topics are gaining attention,
        becoming viral, or may rise next based on
        Reddit discussions.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # SIDEBAR FILTER
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Overview Filters"
    )

    signals = (
        final_ranking[
            "trend_signal"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    signal_options = (
        ["All"]
        + sorted(signals)
    )

    selected_signal = st.sidebar.selectbox(
        "Trend status",
        signal_options
    )


    filtered_ranking = (
        final_ranking.copy()
    )


    if selected_signal != "All":

        filtered_ranking = (
            filtered_ranking[
                filtered_ranking[
                    "trend_signal"
                ]
                == selected_signal
            ]
        )


    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    top_topic = (
        final_ranking
        .sort_values(
            "final_trend_score",
            ascending=False
        )
        .iloc[0]
    )


    most_viral = (
        final_ranking
        .sort_values(
            "virality_score",
            ascending=False
        )
        .iloc[0]
    )


    strongest_forecast = (
        final_ranking
        .sort_values(
            "forecast_strength_score",
            ascending=False
        )
        .iloc[0]
    )


    rising_count = (
        trend_intelligence[
            "predicted_next_day_trend"
        ]
        == "Rising"
    ).sum()


    st.subheader(
        "✨ Today's Key Insights"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "🔥 Top Trend",
            top_topic[
                "topic_name"
            ],
            f"Score {top_topic['final_trend_score']:.2f}"
        )


    with col2:

        st.metric(
            "⚡ Most Viral",
            most_viral[
                "topic_name"
            ],
            f"{most_viral['virality_score']:.2f}"
        )


    with col3:

        st.metric(
            "🔮 Strongest Forecast",
            strongest_forecast[
                "topic_name"
            ],
            f"{strongest_forecast['forecast_strength_score']:.2f}"
        )


    with col4:

        st.metric(
            "📈 Predicted Rising",
            int(rising_count),
            "topics"
        )


    st.divider()


    # --------------------------------------------------------
    # FINAL TREND RANKING
    # --------------------------------------------------------

    st.subheader(
        "🏆 Trend Ranking"
    )

    st.caption(
        "Topics are ranked using current activity, "
        "forecast strength and engagement."
    )


    if filtered_ranking.empty:

        st.warning(
            "No topics match the selected filter."
        )

    else:

        table = (
            filtered_ranking[
                [
                    "final_rank",
                    "topic_name",
                    "trend_signal",
                    "final_trend_score"
                ]
            ]
            .copy()
        )


        table.columns = [
            "Rank",
            "Topic",
            "Trend Status",
            "Trend Score"
        ]


        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # FINAL TREND SCORE
    # --------------------------------------------------------

    if not filtered_ranking.empty:

        st.subheader(
            "⭐ Overall Trend Strength"
        )


        trend_chart = (
            filtered_ranking[
                [
                    "topic_name",
                    "final_trend_score"
                ]
            ]
            .sort_values(
                "final_trend_score",
                ascending=False
            )
            .set_index(
                "topic_name"
            )
        )


        st.bar_chart(
            trend_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CURRENT HEAT VS FORECAST
    # --------------------------------------------------------

    st.subheader(
        "🔥 Current Heat vs 🔮 Forecast Strength"
    )

    st.caption(
        "Compare what is active now with what "
        "the model expects next."
    )


    comparison = (
        final_ranking[
            [
                "topic_name",
                "current_heat_score",
                "forecast_strength_score"
            ]
        ]
        .set_index(
            "topic_name"
        )
    )


    st.bar_chart(
        comparison,
        use_container_width=True
    )


    # --------------------------------------------------------
    # VIRALITY
    # --------------------------------------------------------

    st.subheader(
        "⚡ Most Viral Topics"
    )

    st.caption(
        "Virality represents engagement intensity "
        "rather than discussion volume."
    )


    virality = (
        final_ranking[
            [
                "topic_name",
                "virality_score"
            ]
        ]
        .sort_values(
            "virality_score",
            ascending=False
        )
        .set_index(
            "topic_name"
        )
    )


    st.bar_chart(
        virality,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SCORE EXPLANATIONS
    # --------------------------------------------------------

    with st.expander(
        "ℹ️ How are the scores calculated?"
    ):

        st.markdown(
            """
            ### 🔥 Current Heat

            Indicates how active a topic is now.

            It uses current growth and discussion volume.


            ### 🔮 Forecast Strength

            Indicates how strongly the machine-learning
            model supports the next-day forecast.

            It considers:

            - predicted direction
            - model probability
            - prediction margin


            ### ⚡ Virality

            Measures engagement intensity using:

            - average Reddit score
            - median Reddit score
            - highest Reddit score


            ### ⭐ Final Trend Score

            Combines:

            **80% Trend Intelligence + 20% Virality**

            It is used to rank topics.
            """
        )


        st.warning(
            "Trend Score is a ranking indicator. "
            "It is not the probability that a topic "
            "will become viral."
        )


# ============================================================
# 8. TOPIC EXPLORER PAGE
# ============================================================

  elif page == "🔎 Topic Explorer":

    # --------------------------------------------------------
    # TOPIC SELECTION
    # --------------------------------------------------------

    st.sidebar.subheader(
        "Explore Topic"
    )


    selected_topic = (
        st.sidebar.selectbox(
            "Choose a topic",
            final_ranking[
                "topic_name"
            ].tolist()
        )
    )


    # --------------------------------------------------------
    # GET TOPIC INFORMATION
    # --------------------------------------------------------

    topic_rows = (
        trend_intelligence[
            trend_intelligence[
                "topic_name"
            ]
            == selected_topic
        ]
    )


    if topic_rows.empty:

        st.error(
            "No trend data was found for this topic."
        )

        st.stop()


    topic_info = (
        topic_rows.iloc[0]
    )


    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.title(
        f"🔎 {selected_topic}"
    )

    st.caption(
        "Explore activity, trend prediction, "
        "engagement and the posts driving this topic."
    )


    current_trend = (
        topic_info[
            "current_trend"
        ]
    )

    predicted_trend = (
        topic_info[
            "predicted_next_day_trend"
        ]
    )


    st.subheader(
        f"{trend_icon(current_trend)} "
        f"{current_trend}"
        f"  →  "
        f"{trend_icon(predicted_trend)} "
        f"{predicted_trend}"
    )


    # --------------------------------------------------------
    # TREND SIGNAL
    # --------------------------------------------------------

    signal = (
        topic_info[
            "trend_signal"
        ]
    )


    explanation = (
        signal_explanations.get(
            signal,
            "This topic currently shows mixed trend behaviour."
        )
    )


    st.info(
        f"{signal_icon(signal)} "
        f"**{signal}** — "
        f"{explanation}"
    )


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "🔥 Current Heat",
            f"{topic_info['current_heat_score']:.2f}"
        )


    with col2:

        st.metric(
            "🔮 Forecast Strength",
            f"{topic_info['forecast_strength_score']:.2f}"
        )


    with col3:

        st.metric(
            "⚡ Virality",
            f"{topic_info['virality_score']:.2f}"
        )


    with col4:

        st.metric(
            "⭐ Trend Score",
            f"{topic_info['final_trend_score']:.2f}"
        )


    # --------------------------------------------------------
    # GROWTH RATE
    # --------------------------------------------------------

    growth_rate = (
        topic_info[
            "growth_rate"
        ] * 100
    )


    if growth_rate > 0:

        st.success(
            f"📈 Current discussion growth: "
            f"**+{growth_rate:.2f}%**"
        )

    elif growth_rate < 0:

        st.warning(
            f"📉 Current discussion growth: "
            f"**{growth_rate:.2f}%**"
        )

    else:

        st.info(
            "➡️ Current discussion growth: **0%**"
        )


    # --------------------------------------------------------
    # TREND SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "📊 Trend Snapshot"
    )


    st.write(
        "🔥 **Current Heat**"
    )

    st.progress(
        safe_score(
            topic_info[
                "current_heat_score"
            ]
        ) / 100
    )

    st.caption(
        f"{topic_info['current_heat_score']:.1f} / 100"
    )


    st.write(
        "🔮 **Forecast Strength**"
    )

    st.progress(
        safe_score(
            topic_info[
                "forecast_strength_score"
            ]
        ) / 100
    )

    st.caption(
        f"{topic_info['forecast_strength_score']:.1f} / 100"
    )


    st.write(
        "⚡ **Virality**"
    )

    st.progress(
        safe_score(
            topic_info[
                "virality_score"
            ]
        ) / 100
    )

    st.caption(
        f"{topic_info['virality_score']:.1f} / 100"
    )


    st.write(
        "⭐ **Overall Trend**"
    )

    st.progress(
        safe_score(
            topic_info[
                "final_trend_score"
            ]
        ) / 100
    )

    st.caption(
        f"{topic_info['final_trend_score']:.1f} / 100"
    )


    st.divider()


    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Activity",
            "🤖 Forecast",
            "🔥 Top Posts"
        ]
    )


    # ========================================================
    # TAB 1 — ACTIVITY
    # ========================================================

    with tab1:

        st.subheader(
            "📈 Daily Discussion Activity"
        )


        topic_history = (
            topic_daily[
                topic_daily[
                    "topic_name"
                ]
                == selected_topic
            ]
            .copy()
        )


        topic_history = (
            topic_history
            .sort_values(
                "date"
            )
        )


        if topic_history.empty:

            st.info(
                "No historical activity data is available."
            )

        else:

            history_chart = (
                topic_history[
                    [
                        "date",
                        "post_count"
                    ]
                ]
                .set_index(
                    "date"
                )
            )


            st.line_chart(
                history_chart,
                use_container_width=True
            )


            # Current post count
            latest_posts = (
                topic_history
                .iloc[-1][
                    "post_count"
                ]
            )


            peak_posts = (
                topic_history[
                    "post_count"
                ]
                .max()
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.metric(
                    "Latest Daily Posts",
                    int(latest_posts)
                )


            with col2:

                st.metric(
                    "Peak Daily Posts",
                    int(peak_posts)
                )


        # ----------------------------------------------------
        # INTELLIGENCE COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "🧠 Trend Intelligence"
        )


        intelligence = (
            pd.DataFrame(
                {
                    "Metric": [
                        "Current Heat",
                        "Forecast Strength",
                        "Virality",
                        "Final Trend Score"
                    ],

                    "Score": [
                        topic_info[
                            "current_heat_score"
                        ],

                        topic_info[
                            "forecast_strength_score"
                        ],

                        topic_info[
                            "virality_score"
                        ],

                        topic_info[
                            "final_trend_score"
                        ]
                    ]
                }
            )
            .set_index(
                "Metric"
            )
        )


        st.bar_chart(
            intelligence,
            use_container_width=True
        )


        # ----------------------------------------------------
        # ENGAGEMENT
        # ----------------------------------------------------

        st.subheader(
            "💬 Engagement"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(
                "Average Score",
                f"{topic_info['avg_score']:.2f}"
            )


        with c2:

            st.metric(
                "Median Score",
                f"{topic_info['median_score']:.2f}"
            )


        with c3:

            st.metric(
                "Highest Score",
                f"{topic_info['max_score']:.0f}"
            )


    # ========================================================
    # TAB 2 — FORECAST
    # ========================================================

    with tab2:

        st.subheader(
            "🤖 Next-Day Trend Forecast"
        )


        probability_data = (
            pd.DataFrame(
                {
                    "Trend": [
                        "Rising",
                        "Stable",
                        "Falling"
                    ],

                    "Probability (%)": [
                        topic_info[
                            "Rising_probability"
                        ] * 100,

                        topic_info[
                            "Stable_probability"
                        ] * 100,

                        topic_info[
                            "Falling_probability"
                        ] * 100
                    ]
                }
            )
            .set_index(
                "Trend"
            )
        )


        st.bar_chart(
            probability_data,
            use_container_width=True
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(
                "Prediction",
                predicted_trend
            )


        with c2:

            st.metric(
                "Model Probability",
                f"{topic_info['prediction_probability_percent']:.1f}%"
            )


        with c3:

            st.metric(
                "Prediction Margin",
                f"{topic_info['prediction_margin_percent']:.1f}%"
            )


        # ----------------------------------------------------
        # UNCERTAINTY EXPLANATION
        # ----------------------------------------------------

        margin = (
            topic_info[
                "prediction_margin_percent"
            ]
        )


        if margin < 5:

            st.warning(
                "⚠️ **Very uncertain forecast.** "
                "The top two predicted classes are almost tied."
            )


        elif margin < 10:

            st.info(
                "ℹ️ **Moderately uncertain forecast.** "
                "The model has only a small preference "
                "for the predicted trend."
            )


        else:

            st.success(
                "✅ **Clearer model preference.** "
                "The predicted class has a stronger lead "
                "over the alternatives."
            )


        with st.expander(
            "What does prediction probability mean?"
        ):

            st.write(
                """
                The probability shown here comes from the
                Random Forest model.

                It describes how strongly the model favours
                one class compared with the others.

                It should **not** be interpreted as a guaranteed
                real-world probability because the dataset is
                relatively small and the probabilities have not
                been calibrated.
                """
            )


    # ========================================================
    # TAB 3 — TOP POSTS
    # ========================================================

    with tab3:

        st.subheader(
            "🔥 Posts Driving This Topic"
        )

        st.caption(
            "Highly engaged Reddit posts can help explain "
            "why a topic is receiving attention."
        )


        topic_posts = (
            posts[
                posts[
                    "topic_name"
                ]
                == selected_topic
            ]
            .copy()
        )


        topic_posts = (
            topic_posts
            .sort_values(
                "score",
                ascending=False
            )
            .head(5)
        )


        if topic_posts.empty:

            st.info(
                "No posts are available for this topic."
            )


        else:

            for _, row in topic_posts.iterrows():

                with st.container(
                    border=True
                ):

                    col1, col2 = (
                        st.columns(
                            [1, 4]
                        )
                    )


                    with col1:

                        score = (
                            row.get(
                                "score",
                                0
                            )
                        )

                        if pd.isna(score):
                            score = 0


                        st.metric(
                            "⬆️ Score",
                            f"{score:.0f}"
                        )


                    with col2:

                        subreddit = (
                            row.get(
                                "subreddit",
                                "Unknown"
                            )
                        )

                        st.caption(
                            f"r/{subreddit}"
                        )


                        text = str(
                            row.get(
                                "text",
                                ""
                            )
                        )


                        if len(text) > 700:

                            st.write(
                                text[:700]
                                + "..."
                            )

                        else:

                            st.write(
                                text
                            )


    # ========================================================
    # EXPLANATION SECTION
    # ========================================================

    with st.expander(
        "ℹ️ How should I interpret this topic?"
    ):

        st.markdown(
            """
            **Current Trend**

            Describes what the topic is doing right now.


            **Predicted Next Trend**

            The machine-learning model's next-day forecast.


            **Current Heat**

            Measures current growth and discussion volume.


            **Forecast Strength**

            Measures how strongly the model supports its
            next-day prediction.


            **Virality**

            Measures engagement intensity on Reddit.


            **Final Trend Score**

            Combines trend behaviour and engagement to rank
            the topic against the other topics being analysed.
            """
        )
elif data_source == "New Data Analysis":

    st.title("🔴 New Data Trend Analysis")

    st.write(
        """
        Upload new social-media post data.
        The system will detect topics, calculate trends,
        forecast tomorrow's direction and rank the trends.
        """
    )

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    # Show required format
    with st.expander("📄 Required CSV format"):

        example = pd.DataFrame({
            "created_at": [
                "2026-09-10 09:00",
                "2026-09-11 11:30"
            ],

            "text": [
                "Qwen released a new reasoning model.",
                "Running Llama locally using Ollama."
            ],

            "score": [
                120,
                75
            ]
        })

        st.dataframe(
            example,
            use_container_width=True,
            hide_index=True
        )


    # -----------------------------------------------------
    # WHEN USER UPLOADS FILE
    # -----------------------------------------------------

    if uploaded_file is not None:

        new_data = pd.read_csv(
            uploaded_file
        )

        required_columns = {
            "created_at",
            "text",
            "score"
        }

        missing_columns = (
            required_columns
            - set(new_data.columns)
        )


        # -------------------------------------------------
        # VALIDATE COLUMNS
        # -------------------------------------------------

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.stop()


        # -------------------------------------------------
        # BASIC INFORMATION
        # -------------------------------------------------

        st.success(
            f"Loaded {len(new_data):,} posts successfully."
        )

        temp_dates = pd.to_datetime(
            new_data["created_at"],
            errors="coerce"
        )

        number_of_days = (
            temp_dates.dt.date.nunique()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Posts",
            len(new_data)
        )

        col2.metric(
            "Days",
            number_of_days
        )

        col3.metric(
            "Average Score",
            round(
                new_data["score"].mean(),
                1
            )
        )


        # -------------------------------------------------
        # RUN OUR ML PIPELINE
        # -------------------------------------------------

        with st.spinner(
            "Analysing posts..."
        ):

            results = analyse_posts(
                new_data
            )


        # -------------------------------------------------
        # GET RESULTS
        # -------------------------------------------------

        classified_posts = (
            results["classified_posts"]
        )

        growth_data = (
            results["growth_data"]
        )

        forecast = (
            results["forecast"]
        )

        # THIS IS THE NEW LINE FOR RANKING
        final_ranking = (
            results["final_ranking"]
        )


        # =================================================
        # 🔥 FINAL TREND RANKING
        # =================================================

        st.subheader(
            "🔥 Trend Ranking"
        )

        if final_ranking.empty:

            st.warning(
                "There is not enough historical activity "
                "to create a trend ranking yet."
            )

        else:

            # ---------------------------------------------
            # TOP TREND CARD
            # ---------------------------------------------

            top = final_ranking.iloc[0]

            st.success(
                f"🔥 Top Trend: "
                f"{top['topic_name']} — "
                f"{top['trend_signal']} "
                f"(Score: {top['final_trend_score']:.1f})"
            )


            # ---------------------------------------------
            # RANKING TABLE
            # ---------------------------------------------

            ranking_display = final_ranking[
                [
                    "rank",
                    "topic_name",
                    "trend_signal",
                    "current_heat_score",
                    "forecast_strength_score",
                    "virality_score",
                    "final_trend_score"
                ]
            ].copy()


            score_columns = [
                "current_heat_score",
                "forecast_strength_score",
                "virality_score",
                "final_trend_score"
            ]

            ranking_display[
                score_columns
            ] = (
                ranking_display[
                    score_columns
                ].round(1)
            )


            ranking_display.columns = [
                "Rank",
                "Topic",
                "Trend Status",
                "Current Heat",
                "Forecast Strength",
                "Virality",
                "Trend Score"
            ]


            st.dataframe(
                ranking_display,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------------
            # RANKING CHART
            # ---------------------------------------------

            ranking_chart = (
                final_ranking[
                    [
                        "topic_name",
                        "final_trend_score"
                    ]
                ]
                .set_index(
                    "topic_name"
                )
            )

            st.bar_chart(
                ranking_chart
            )


        # =================================================
        # 🧠 DETECTED TOPICS
        # =================================================

        st.subheader(
            "🧠 Detected Topics"
        )

        topic_counts = (
            classified_posts[
                "topic_name"
            ]
            .value_counts()
            .reset_index()
        )

        topic_counts.columns = [
            "Topic",
            "Posts"
        ]

        st.dataframe(
            topic_counts,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # 📈 DAILY ACTIVITY
        # =================================================

        st.subheader(
            "📈 Topic Activity Over Time"
        )

        chart_data = (
            growth_data.pivot(
                index="date",
                columns="topic_name",
                values="post_count"
            )
        )

        st.line_chart(
            chart_data
        )


        # =================================================
        # 🔮 FORECAST
        # =================================================

        st.subheader(
            "🔮 Next-Day Trend Forecast"
        )

        if forecast.empty:

            st.warning(
                "There is not enough historical data "
                "to create a forecast."
            )

        else:

            forecast_display = forecast[
                [
                    "topic_name",
                    "current_trend",
                    "predicted_next_day_trend",
                    "prediction_probability",
                    "prediction_margin"
                ]
            ].copy()


            forecast_display[
                "prediction_probability"
            ] = (
                forecast_display[
                    "prediction_probability"
                ] * 100
            ).round(1)


            forecast_display[
                "prediction_margin"
            ] = (
                forecast_display[
                    "prediction_margin"
                ] * 100
            ).round(1)


            forecast_display.columns = [
                "Topic",
                "Current Trend",
                "Predicted Tomorrow",
                "Model Probability (%)",
                "Prediction Margin (%)"
            ]


            st.dataframe(
                forecast_display,
                use_container_width=True,
                hide_index=True
            )
st.divider()

st.caption(
    "AI-Powered Social Media Trend Analyser • "
    "Reddit AI trend analysis prototype"
)