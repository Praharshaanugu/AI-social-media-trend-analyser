from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# =========================================================
# LOAD SAVED MODELS
# =========================================================

topic_vectorizer = joblib.load(
    MODELS_DIR / "topic_vectorizer.joblib"
)

nmf_model = joblib.load(
    MODELS_DIR / "nmf_topic_model.joblib"
)

trend_forecaster = joblib.load(
    MODELS_DIR / "trend_forecaster.joblib"
)


with open(
    MODELS_DIR / "feature_columns.json",
    "r"
) as f:
    feature_columns = json.load(f)


with open(
    MODELS_DIR / "topic_names.json",
    "r"
) as f:
    topic_names = json.load(f)

topic_names = {
    int(k): v
    for k, v in topic_names.items()
}


with open(
    MODELS_DIR / "project_config.json",
    "r"
) as f:
    project_config = json.load(f)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    text = re.sub(
        r'\[([^\]]+)\]\([^)]+\)',
        r'\1',
        text
    )

    text = re.sub(
        r'[\*_\~`]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# =========================================================
# TOPIC CLASSIFICATION
# =========================================================

def classify_posts(df):

    data = df.copy()

    data["clean_text"] = (
        data["text"]
        .fillna("")
        .apply(clean_text)
    )

    X = topic_vectorizer.transform(
        data["clean_text"]
    )

    W = nmf_model.transform(X)

    data["topic"] = W.argmax(axis=1)

    data["topic_strength"] = (
        W.max(axis=1)
    )

    data["topic_name"] = (
        data["topic"]
        .map(topic_names)
    )

    return data


# =========================================================
# DAILY TOPIC AGGREGATION
# =========================================================

def aggregate_daily_topics(df):

    data = df.copy()

    data["created_at"] = pd.to_datetime(
        data["created_at"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["created_at"]
    )

    data["date"] = (
        data["created_at"]
        .dt.normalize()
    )

    daily = (
        data.groupby(
            [
                "date",
                "topic",
                "topic_name"
            ]
        )
        .agg(
            post_count=("text", "count")
        )
        .reset_index()
    )

    return daily


# =========================================================
# COMPLETE DATE × TOPIC GRID
# =========================================================

def complete_topic_grid(daily_df):

    df = daily_df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    all_dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    all_topics = list(
        topic_names.keys()
    )

    full_index = (
        pd.MultiIndex.from_product(
            [
                all_dates,
                all_topics
            ],
            names=[
                "date",
                "topic"
            ]
        )
    )

    complete = (
        df
        .set_index(
            ["date", "topic"]
        )
        .reindex(full_index)
        .reset_index()
    )

    complete["post_count"] = (
        complete["post_count"]
        .fillna(0)
        .astype(int)
    )

    complete["topic_name"] = (
        complete["topic"]
        .map(topic_names)
    )

    return complete


# =========================================================
# DAILY GROWTH
# =========================================================

def add_growth_features(df):

    data = (
        df.copy()
        .sort_values(
            ["topic", "date"]
        )
        .reset_index(drop=True)
    )

    data["previous_post_count"] = (
        data
        .groupby("topic")[
            "post_count"
        ]
        .shift(1)
    )

    data["growth"] = (
        data["post_count"]
        -
        data["previous_post_count"]
    )

    data["growth_rate"] = np.where(
        data["previous_post_count"] > 0,

        data["growth"]
        /
        data["previous_post_count"],

        np.nan
    )

    data["trend"] = data.apply(
        assign_trend,
        axis=1
    )

    return data


def assign_trend(row):

    current = row["post_count"]
    previous = row["previous_post_count"]
    rate = row["growth_rate"]

    if pd.isna(previous):
        return "Unknown"

    if previous == 0 and current == 0:
        return "Stable"

    if previous == 0 and current > 0:
        return "Rising"

    if rate >= 0.20:
        return "Rising"

    if rate <= -0.20:
        return "Falling"

    return "Stable"


# =========================================================
# RANDOM FOREST FEATURES
# =========================================================

def build_forecast_features(df):

    latest_date = df["date"].max()

    latest = df[
        df["date"] == latest_date
    ].copy()

    latest = latest[
        latest["growth_rate"].notna()
    ].copy()

    latest["day_of_week"] = (
        latest["date"]
        .dt.dayofweek
    )

    for trend in [
        "Falling",
        "Rising",
        "Stable"
    ]:

        latest[
            f"trend_{trend}"
        ] = (
            latest["trend"] == trend
        ).astype(int)

    for topic_id in range(8):

        latest[
            f"topic_{topic_id}"
        ] = (
            latest["topic"]
            == topic_id
        ).astype(int)

    X = latest[
        feature_columns
    ].copy()

    return latest, X
def forecast_trends(growth_df):

    latest, X = build_forecast_features(
        growth_df
    )

    if X.empty:
        return pd.DataFrame()

    predictions = (
        trend_forecaster.predict(X)
    )

    probabilities = (
        trend_forecaster.predict_proba(X)
    )

    sorted_probs = np.sort(
        probabilities,
        axis=1
    )

    result = latest[
        [
            "date",
            "topic",
            "topic_name",
            "post_count",
            "previous_post_count",
            "growth",
            "growth_rate",
            "trend"
        ]
    ].copy()

    result = result.rename(
        columns={
            "trend": "current_trend"
        }
    )

    result[
        "predicted_next_day_trend"
    ] = predictions

    result[
        "prediction_probability"
    ] = probabilities.max(axis=1)

    result[
        "prediction_margin"
    ] = (
        sorted_probs[:, -1]
        - sorted_probs[:, -2]
    )

    result["forecast_for"] = (
        result["date"]
        + pd.Timedelta(days=1)
    )

    return result
def get_trend_signal(current, predicted):

    transitions = {
        ("Stable", "Rising"): "Emerging Trend",
        ("Falling", "Rising"): "Possible Rebound",
        ("Rising", "Rising"): "Continuing Rise",

        ("Rising", "Stable"): "Stabilizing",
        ("Stable", "Stable"): "Steady",
        ("Falling", "Stable"): "Recovering / Stabilizing",

        ("Rising", "Falling"): "Cooling Down",
        ("Stable", "Falling"): "Possible Decline",
        ("Falling", "Falling"): "Continuing Decline"
    }

    return transitions.get(
        (current, predicted),
        "Unknown"
    )
def minmax_100(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
        * 100
    )
def build_trend_intelligence(
    classified_posts,
    forecast_df
):

    if forecast_df.empty:
        return pd.DataFrame()

    result = forecast_df.copy()

    # =====================================================
    # TREND SIGNAL
    # =====================================================

    result["trend_signal"] = result.apply(
        lambda row: get_trend_signal(
            row["current_trend"],
            row["predicted_next_day_trend"]
        ),
        axis=1
    )


    # =====================================================
    # CURRENT HEAT
    # =====================================================

    result["growth_score"] = (
        result["growth_rate"]
        .rank(pct=True)
        * 100
    )

    result["volume_score"] = (
        result["post_count"]
        .rank(pct=True)
        * 100
    )

    heat_weights = (
        project_config[
            "current_heat_weights"
        ]
    )

    result["current_heat_score"] = (
        heat_weights["growth"]
        * result["growth_score"]

        +

        heat_weights["volume"]
        * result["volume_score"]
    )


    # =====================================================
    # FORECAST STRENGTH
    # =====================================================

    direction_scores = {
        "Rising": 100,
        "Stable": 50,
        "Falling": 0
    }

    result["direction_score"] = (
        result[
            "predicted_next_day_trend"
        ]
        .map(direction_scores)
    )

    result[
        "prediction_probability_pct"
    ] = (
        result[
            "prediction_probability"
        ] * 100
    )

    result[
        "prediction_margin_pct"
    ] = (
        result[
            "prediction_margin"
        ] * 100
    )

    forecast_weights = (
        project_config[
            "forecast_strength_weights"
        ]
    )

    result[
        "forecast_strength_score"
    ] = (

        forecast_weights["direction"]
        * result["direction_score"]

        +

        forecast_weights[
            "prediction_probability"
        ]
        * result[
            "prediction_probability_pct"
        ]

        +

        forecast_weights[
            "prediction_margin"
        ]
        * result[
            "prediction_margin_pct"
        ]
    )


    # =====================================================
    # OVERALL TREND SCORE
    # =====================================================

    intelligence_weights = (
        project_config[
            "trend_intelligence_weights"
        ]
    )

    result[
        "overall_trend_score"
    ] = (

        intelligence_weights[
            "current_heat"
        ]
        * result[
            "current_heat_score"
        ]

        +

        intelligence_weights[
            "forecast_strength"
        ]
        * result[
            "forecast_strength_score"
        ]
    )


    # =====================================================
    # LATEST-DAY ENGAGEMENT
    # =====================================================

    posts = classified_posts.copy()

    posts["created_at"] = pd.to_datetime(
        posts["created_at"],
        errors="coerce"
    )

    posts["score"] = pd.to_numeric(
        posts["score"],
        errors="coerce"
    ).fillna(0)

    posts = posts.dropna(
        subset=["created_at"]
    )

    posts["date"] = (
        posts["created_at"]
        .dt.normalize()
    )

    latest_date = posts["date"].max()

    latest_posts = posts[
        posts["date"] == latest_date
    ].copy()

    engagement = (
        latest_posts
        .groupby(
            ["topic", "topic_name"]
        )
        .agg(
            engagement_posts=(
                "text",
                "count"
            ),

            total_score=(
                "score",
                "sum"
            ),

            avg_score=(
                "score",
                "mean"
            ),

            median_score=(
                "score",
                "median"
            ),

            max_score=(
                "score",
                "max"
            )
        )
        .reset_index()
    )


    # =====================================================
    # VIRALITY
    # =====================================================

    if not engagement.empty:

        for column in [
            "avg_score",
            "median_score",
            "max_score"
        ]:

            engagement[
                f"log_{column}"
            ] = np.log1p(
                engagement[column]
            )

        engagement[
            "avg_engagement_score"
        ] = minmax_100(
            engagement["log_avg_score"]
        )

        engagement[
            "median_engagement_score"
        ] = minmax_100(
            engagement["log_median_score"]
        )

        engagement[
            "max_engagement_score"
        ] = minmax_100(
            engagement["log_max_score"]
        )

        viral_weights = (
            project_config[
                "virality_weights"
            ]
        )

        engagement[
            "virality_score"
        ] = (

            viral_weights[
                "average_score"
            ]
            * engagement[
                "avg_engagement_score"
            ]

            +

            viral_weights[
                "median_score"
            ]
            * engagement[
                "median_engagement_score"
            ]

            +

            viral_weights[
                "maximum_score"
            ]
            * engagement[
                "max_engagement_score"
            ]
        )

    else:

        engagement = pd.DataFrame(
            columns=[
                "topic",
                "engagement_posts",
                "total_score",
                "avg_score",
                "median_score",
                "max_score",
                "virality_score"
            ]
        )


    # =====================================================
    # MERGE ENGAGEMENT
    # =====================================================

    result = result.merge(
        engagement[
            [
                "topic",
                "engagement_posts",
                "total_score",
                "avg_score",
                "median_score",
                "max_score",
                "virality_score"
            ]
        ],
        on="topic",
        how="left"
    )

    result[
        "virality_score"
    ] = (
        result[
            "virality_score"
        ]
        .fillna(0)
    )


    # =====================================================
    # FINAL SCORE
    # =====================================================

    final_weights = (
        project_config[
            "final_score_weights"
        ]
    )

    result[
        "final_trend_score"
    ] = (

        final_weights[
            "trend_intelligence"
        ]
        * result[
            "overall_trend_score"
        ]

        +

        final_weights[
            "virality"
        ]
        * result[
            "virality_score"
        ]
    )


    # =====================================================
    # FINAL RANKING
    # =====================================================

    result = (
        result
        .sort_values(
            "final_trend_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    result["rank"] = (
        result.index + 1
    )

    return result
def analyse_posts(df):

    classified = classify_posts(df)

    daily = aggregate_daily_topics(
        classified
    )

    complete = complete_topic_grid(
        daily
    )

    growth = add_growth_features(
        complete
    )

    forecast = forecast_trends(
        growth
    )

    ranking = build_trend_intelligence(
        classified,
        forecast
    )

    return {
        "classified_posts": classified,
        "daily_topics": daily,
        "complete_daily": complete,
        "growth_data": growth,
        "forecast": forecast,
        "final_ranking": ranking
    }