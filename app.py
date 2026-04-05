from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="WeatherSphere Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_DATA_PATH = Path(r"D:\desktop\weather analysis infosys\GlobalWeatherRepository .csv")
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
AQI_COL = "air_quality_PM2.5"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #09121d;
            --panel: rgba(10, 25, 40, 0.74);
            --panel-2: rgba(20, 44, 66, 0.78);
            --line: rgba(177, 219, 255, 0.16);
            --text: #e7f4ff;
            --muted: #97b7cf;
            --accent: #6ae3ff;
            --accent-2: #ffb85c;
            --accent-3: #7dffb3;
            --danger: #ff7f8f;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(106, 227, 255, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(255, 184, 92, 0.16), transparent 28%),
                linear-gradient(135deg, #071019 0%, #0d1c2b 44%, #12273a 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 17, 28, 0.98), rgba(10, 22, 35, 0.98));
            border-right: 1px solid var(--line);
        }

        .hero {
            padding: 1.5rem 1.6rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
                linear-gradient(145deg, rgba(12, 31, 47, 0.95), rgba(9, 20, 31, 0.82)),
                radial-gradient(circle at top right, rgba(106, 227, 255, 0.18), transparent 30%);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.24);
            margin-bottom: 1rem;
        }

        .hero h1, .hero h3, .hero p {
            color: var(--text);
            margin: 0;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 1rem;
            align-items: center;
        }

        .glass {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(8px);
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(14, 32, 49, 0.88), rgba(9, 22, 34, 0.82));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            min-height: 118px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        .metric-sub {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.7rem 1rem;
        }

        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.3rem;
            font-size: 1.1rem;
            color: var(--text);
            font-weight: 700;
        }

        .section-subtitle {
            color: var(--muted);
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def season_from_month(month: int) -> str:
    return SEASON_MAP.get(month, "Unknown")


def normalize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    spread = series.max() - series.min()
    if np.isclose(spread, 0):
        return pd.Series(50.0, index=series.index)
    return ((series - series.min()) / spread) * 100


def comfort_from_temp(temp_c: pd.Series) -> pd.Series:
    comfort = 100 - (temp_c.astype(float).sub(22).abs() * 4.2)
    return comfort.clip(0, 100)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file, csv_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path)

    df = df.copy()
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df = df.dropna(subset=["last_updated"]).drop_duplicates()
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    df["day"] = df["last_updated"].dt.day
    df["hour"] = df["last_updated"].dt.hour
    df["season"] = df["month"].map(season_from_month)
    df["temp_deviation"] = df["temperature_celsius"] - df.groupby("location_name")["temperature_celsius"].transform("median")
    df["precip_deviation"] = df["precip_mm"] - df.groupby("location_name")["precip_mm"].transform("median")
    df["comfort_score"] = (
        comfort_from_temp(df["temperature_celsius"])
        - (df["humidity"].astype(float) - 55).abs() * 0.45
        - df["wind_kph"].astype(float) * 0.18
        - df["precip_mm"].astype(float) * 1.5
    ).clip(0, 100)
    df["visibility_score"] = normalize(df["visibility_km"])
    df["air_quality_score"] = (100 - normalize(df[AQI_COL].fillna(df[AQI_COL].median()))).clip(0, 100)
    df["travel_readiness"] = (
        df["comfort_score"] * 0.48
        + df["visibility_score"] * 0.22
        + df["air_quality_score"] * 0.2
        + (100 - normalize(df["cloud"])) * 0.1
    ).clip(0, 100)
    return df


@st.cache_data(show_spinner=False)
def build_climate_clusters(df: pd.DataFrame, n_clusters: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = ["temperature_celsius", "humidity", "wind_kph", "precip_mm", AQI_COL]
    work = df[["location_name", "country"] + feature_cols].dropna().copy()
    city_profile = work.groupby(["location_name", "country"], as_index=False)[feature_cols].median()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(city_profile[feature_cols])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    city_profile["cluster"] = model.fit_predict(scaled)

    cluster_summary = city_profile.groupby("cluster")[feature_cols].mean().round(2)
    cluster_summary["profile_name"] = cluster_summary.apply(label_cluster, axis=1)
    city_profile = city_profile.merge(
        cluster_summary["profile_name"].rename("climate_persona"),
        left_on="cluster",
        right_index=True,
        how="left",
    )
    return city_profile, cluster_summary.reset_index()


def label_cluster(row: pd.Series) -> str:
    temp = row["temperature_celsius"]
    humidity = row["humidity"]
    precip = row["precip_mm"]
    wind = row["wind_kph"]
    pm25 = row[AQI_COL]

    if precip > 3.5 and humidity > 72:
        return "Rain-Lush Belt"
    if temp > 28 and humidity < 45:
        return "Dry Heat Zone"
    if temp < 14 and wind > 15:
        return "Cool Wind Corridor"
    if pm25 > 55:
        return "Dense Urban Air"
    return "Balanced Climate Hub"


def apply_filters(
    df: pd.DataFrame,
    countries: Iterable[str],
    locations: Iterable[str],
    seasons: Iterable[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if countries:
        mask &= df["country"].isin(countries)
    if locations:
        mask &= df["location_name"].isin(locations)
    if seasons:
        mask &= df["season"].isin(seasons)
    start_date, end_date = date_range
    mask &= df["last_updated"].between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1))
    return df.loc[mask].copy()


def build_similarity_table(city_profile: pd.DataFrame, selected_city: str) -> pd.DataFrame:
    feature_cols = ["temperature_celsius", "humidity", "wind_kph", "precip_mm", AQI_COL]
    if selected_city not in set(city_profile["location_name"]):
        return pd.DataFrame()

    base_row = city_profile.loc[city_profile["location_name"] == selected_city, feature_cols]
    distances = euclidean_distances(base_row, city_profile[feature_cols])[0]
    out = city_profile[["location_name", "country", "climate_persona"]].copy()
    out["distance"] = distances
    out["match_score"] = (100 - normalize(pd.Series(distances))).round(1)
    out = out[out["location_name"] != selected_city].sort_values(["distance", "location_name"]).head(8)
    return out


def simple_forecast(city_df: pd.DataFrame, periods: int = 8) -> pd.DataFrame:
    series = (
        city_df.sort_values("last_updated")
        .set_index("last_updated")["temperature_celsius"]
        .resample("D")
        .mean()
        .dropna()
    )
    if len(series) < 5:
        return pd.DataFrame()

    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, deg=1)
    future_x = np.arange(len(series), len(series) + periods)
    y_future = coeffs[0] * future_x + coeffs[1]
    future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=periods, freq="D")
    return pd.DataFrame({"last_updated": future_dates, "forecast_temp_c": y_future})


def format_big_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:.0f}"


def metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(filtered_df: pd.DataFrame, city_profile: pd.DataFrame) -> None:
    latest_ts = filtered_df["last_updated"].max()
    hottest = filtered_df.loc[filtered_df["temperature_celsius"].idxmax()]
    cleanest = filtered_df.loc[filtered_df[AQI_COL].idxmin()]
    ready = filtered_df.loc[filtered_df["travel_readiness"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Records", format_big_number(len(filtered_df)), f"{filtered_df['country'].nunique()} countries in view")
    with c2:
        metric_card("Average Temp", f"{filtered_df['temperature_celsius'].mean():.1f}°C", "Across active filters")
    with c3:
        metric_card("Travel Readiness", f"{filtered_df['travel_readiness'].mean():.0f}/100", "Comfort + visibility + AQI blend")
    with c4:
        metric_card("Last Observation", latest_ts.strftime("%d %b %Y %H:%M"), "Latest timestamp in filtered data")

    row1, row2 = st.columns((1.45, 1))
    with row1:
        st.markdown('<div class="section-title">Global Weather Pulse</div>', unsafe_allow_html=True)
        trend = (
            filtered_df.groupby(pd.Grouper(key="last_updated", freq="D"))[
                ["temperature_celsius", "humidity", "precip_mm"]
            ]
            .mean()
            .reset_index()
            .dropna()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["temperature_celsius"], mode="lines", name="Temperature (°C)", line=dict(color="#6ae3ff", width=3)))
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["humidity"], mode="lines", name="Humidity (%)", line=dict(color="#ffb85c", width=2)))
        fig.add_trace(go.Bar(x=trend["last_updated"], y=trend["precip_mm"], name="Precipitation (mm)", marker_color="rgba(125,255,179,0.45)"))
        fig.update_layout(height=430, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with row2:
        st.markdown('<div class="section-title">Signal Highlights</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass">
                <p><strong>Hottest observed point:</strong> {hottest['location_name']}, {hottest['country']} at {hottest['temperature_celsius']:.1f}°C.</p>
                <p><strong>Cleanest air pocket:</strong> {cleanest['location_name']}, {cleanest['country']} with PM2.5 at {cleanest[AQI_COL]:.1f}.</p>
                <p><strong>Best outdoor profile:</strong> {ready['location_name']}, {ready['country']} scored {ready['travel_readiness']:.0f}/100.</p>
                <p><strong>Dominant climate persona:</strong> {city_profile['climate_persona'].mode().iloc[0]}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        season_mix = filtered_df["season"].value_counts().reset_index()
        season_mix.columns = ["season", "count"]
        donut = px.pie(
            season_mix,
            names="season",
            values="count",
            hole=0.58,
            color_discrete_sequence=["#6ae3ff", "#7dffb3", "#ffb85c", "#ff7f8f", "#aab7c4"],
        )
        donut.update_layout(height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(donut, use_container_width=True)

    row3, row4 = st.columns(2)
    with row3:
        top_ready = (
            filtered_df.groupby(["location_name", "country"], as_index=False)["travel_readiness"]
            .mean()
            .sort_values("travel_readiness", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_ready,
            x="travel_readiness",
            y="location_name",
            color="travel_readiness",
            orientation="h",
            hover_data=["country"],
            color_continuous_scale=["#18415f", "#6ae3ff", "#ffb85c"],
            title="Top Outdoor-Friendly Cities",
        )
        fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with row4:
        avg_by_month = (
            filtered_df.groupby("month", as_index=False)["temperature_celsius"]
            .mean()
            .sort_values("month")
        )
        fig = px.area(
            avg_by_month,
            x="month",
            y="temperature_celsius",
            markers=True,
            title="Seasonal Temperature Rhythm",
            color_discrete_sequence=["#6ae3ff"],
        )
        fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def geo_tab(filtered_df: pd.DataFrame) -> None:
    left, right = st.columns((1.2, 1))
    with left:
        st.markdown('<div class="section-title">Global Temperature Atlas</div>', unsafe_allow_html=True)
        country_temp = filtered_df.groupby("country", as_index=False)["temperature_celsius"].mean()
        fig = px.choropleth(
            country_temp,
            locations="country",
            locationmode="country names",
            color="temperature_celsius",
            color_continuous_scale="Turbo",
            hover_name="country",
        )
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Live Conditions Map</div>', unsafe_allow_html=True)
        sample_size = min(len(filtered_df), 3500)
        sampled = filtered_df.sample(sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df
        fig = px.scatter_geo(
            sampled,
            lat="latitude",
            lon="longitude",
            color="temperature_celsius",
            size="humidity",
            hover_name="location_name",
            hover_data=["country", "condition_text", "temperature_celsius", "humidity", "wind_kph"],
            color_continuous_scale="IceFire",
            projection="natural earth",
        )
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        rain = (
            filtered_df.groupby("country", as_index=False)["precip_mm"]
            .mean()
            .sort_values("precip_mm", ascending=False)
            .head(12)
        )
        fig = px.bar(
            rain,
            x="country",
            y="precip_mm",
            color="precip_mm",
            title="Rainfall Hotspots",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with lower_right:
        air = (
            filtered_df.groupby("country", as_index=False)[AQI_COL]
            .mean()
            .sort_values(AQI_COL, ascending=False)
            .head(12)
        )
        fig = px.bar(
            air,
            x=AQI_COL,
            y="country",
            color=AQI_COL,
            orientation="h",
            title="Air Quality Pressure",
            color_continuous_scale="OrRd",
        )
        fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def climate_lab_tab(filtered_df: pd.DataFrame, city_profile: pd.DataFrame, cluster_summary: pd.DataFrame) -> None:
    left, right = st.columns((1.2, 1))
    with left:
        st.markdown('<div class="section-title">Climate Persona Map</div>', unsafe_allow_html=True)
        fig = px.scatter(
            city_profile,
            x="temperature_celsius",
            y="humidity",
            color="climate_persona",
            size="wind_kph",
            hover_data=["location_name", "country", "precip_mm", AQI_COL],
            title="Clustered city signatures",
            color_discrete_sequence=["#6ae3ff", "#ffb85c", "#7dffb3", "#ff7f8f", "#c2b6ff"],
        )
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Persona Breakdown</div>', unsafe_allow_html=True)
        summary_view = cluster_summary.rename(
            columns={
                "temperature_celsius": "Temp (C)",
                "humidity": "Humidity",
                "wind_kph": "Wind",
                "precip_mm": "Rain",
                AQI_COL: "PM2.5",
                "profile_name": "Persona",
            }
        )[["Persona", "Temp (C)", "Humidity", "Wind", "Rain", "PM2.5"]]
        st.dataframe(summary_view, use_container_width=True, hide_index=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown('<div class="section-title">Anomaly Tracker</div>', unsafe_allow_html=True)
        anomaly_df = filtered_df.copy()
        anomaly_df["temp_z"] = anomaly_df.groupby("location_name")["temperature_celsius"].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() and not np.isnan(s.std()) else 1)
        )
        anomaly_df = anomaly_df[anomaly_df["temp_z"].abs() >= 2.2].sort_values("last_updated")
        if anomaly_df.empty:
            st.info("No strong temperature anomalies were found in the current filtered window.")
        else:
            timeline = anomaly_df.groupby(pd.Grouper(key="last_updated", freq="D")).size().reset_index(name="count")
            fig = px.line(timeline, x="last_updated", y="count", markers=True, title="Extreme temperature events")
            fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with lower_right:
        st.markdown('<div class="section-title">Condition Matrix</div>', unsafe_allow_html=True)
        matrix = filtered_df[["temperature_celsius", "humidity", "wind_kph", "precip_mm", AQI_COL, "visibility_km"]].corr().round(2)
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def comparison_tab(filtered_df: pd.DataFrame, city_profile: pd.DataFrame) -> None:
    available_cities = sorted(filtered_df["location_name"].unique())
    if not available_cities:
        st.warning("No cities are available for comparison with the current filters.")
        return

    default_cities = available_cities[: min(3, len(available_cities))]
    selected = st.multiselect("Select cities to compare", available_cities, default=default_cities)
    if not selected:
        st.info("Choose at least one city to unlock the comparison dashboard.")
        return

    compare_df = (
        filtered_df[filtered_df["location_name"].isin(selected)]
        .groupby(["location_name", "country"], as_index=False)[
            ["temperature_celsius", "humidity", "wind_kph", "precip_mm", "visibility_km", "travel_readiness", AQI_COL]
        ]
        .mean()
    )

    left, right = st.columns((1.15, 1))
    with left:
        fig = go.Figure()
        radar_cols = ["temperature_celsius", "humidity", "wind_kph", "visibility_km", "travel_readiness"]
        for _, row in compare_df.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[row[c] for c in radar_cols],
                    theta=["Temp", "Humidity", "Wind", "Visibility", "Travel Readiness"],
                    fill="toself",
                    name=row["location_name"],
                )
            )
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", polar=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">City Scoreboard</div>', unsafe_allow_html=True)
        table = compare_df.copy()
        table["AQI Risk"] = pd.cut(table[AQI_COL], bins=[-1, 12, 35, 55, 150, np.inf], labels=["Excellent", "Fair", "Sensitive", "High", "Severe"])
        st.dataframe(
            table.rename(
                columns={
                    "location_name": "City",
                    "country": "Country",
                    "temperature_celsius": "Temp (C)",
                    "humidity": "Humidity",
                    "wind_kph": "Wind",
                    "precip_mm": "Rain",
                    "visibility_km": "Visibility",
                    "travel_readiness": "Travel Score",
                    AQI_COL: "PM2.5",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    focus_city = st.selectbox("Find weather twins for", available_cities, index=0)
    twins = build_similarity_table(city_profile, focus_city)
    st.markdown('<div class="section-title">Weather Twin Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Cities with the most similar climate signature based on temperature, humidity, wind, rainfall, and PM2.5.</div>', unsafe_allow_html=True)
    st.dataframe(
        twins.rename(
            columns={
                "location_name": "City",
                "country": "Country",
                "climate_persona": "Persona",
                "distance": "Distance",
                "match_score": "Match Score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def explorer_tab(filtered_df: pd.DataFrame) -> None:
    left, right = st.columns((1.15, 1))
    with left:
        st.markdown('<div class="section-title">Local Forecast Sandbox</div>', unsafe_allow_html=True)
        forecast_city = st.selectbox("Choose a city for trend projection", sorted(filtered_df["location_name"].unique()), index=0, key="forecast_city")
        city_df = filtered_df[filtered_df["location_name"] == forecast_city].copy()
        forecast_df = simple_forecast(city_df)
        history = city_df.sort_values("last_updated").set_index("last_updated")["temperature_celsius"].resample("D").mean().dropna().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["last_updated"], y=history["temperature_celsius"], mode="lines+markers", name="History", line=dict(color="#6ae3ff", width=3)))
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(x=forecast_df["last_updated"], y=forecast_df["forecast_temp_c"], mode="lines+markers", name="Projection", line=dict(color="#ffb85c", width=3, dash="dash")))
        fig.update_layout(height=420, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        if forecast_df.empty:
            st.caption("Not enough daily history in the current filter window to build a simple projection for this city.")

    with right:
        st.markdown('<div class="section-title">Best Time Window</div>', unsafe_allow_html=True)
        hourly = (
            filtered_df.groupby("hour", as_index=False)[["comfort_score", "travel_readiness"]]
            .mean()
            .sort_values("travel_readiness", ascending=False)
        )
        best_hour = int(hourly.iloc[0]["hour"])
        st.markdown(
            f"""
            <div class="glass">
                <p><strong>Recommended hour:</strong> {best_hour:02d}:00 local dataset time.</p>
                <p><strong>Why it stands out:</strong> it combines higher comfort, better visibility, and calmer weather than the surrounding hours.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = px.bar(hourly.sort_values("hour"), x="hour", y="travel_readiness", color="comfort_score", color_continuous_scale="Tealgrn")
        fig.update_layout(height=320, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Filtered Dataset</div>', unsafe_allow_html=True)
    preview_cols = [
        "country",
        "location_name",
        "last_updated",
        "temperature_celsius",
        "condition_text",
        "humidity",
        "wind_kph",
        "precip_mm",
        AQI_COL,
        "travel_readiness",
    ]
    st.dataframe(filtered_df[preview_cols].sort_values("last_updated", ascending=False), use_container_width=True, height=330)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="weather_intelligence_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <h1>WeatherSphere Intelligence</h1>
                    <p style="margin-top:0.8rem; color:#97b7cf; font-size:1.02rem;">
                        An advanced weather analytics web app built from your global weather repository.
                        Explore live climate patterns, air quality pressure, anomaly spikes, city-level comparisons,
                        and a custom travel-readiness engine in one polished dashboard.
                    </p>
                </div>
                <div class="glass">
                    <h3>Unique Features</h3>
                    <p style="color:#97b7cf; margin-top:0.6rem;">
                        Climate personas, weather twins, anomaly tracking, outdoor scorecards,
                        forecast sandbox, and export-ready filtered insights.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Control Center")
        uploaded_file = st.file_uploader("Upload a weather CSV", type=["csv"])
        csv_path = st.text_input("Or use local CSV path", value=str(DEFAULT_DATA_PATH))

    try:
        df = load_data(uploaded_file, csv_path)
    except Exception as exc:
        st.error(f"Unable to load the dataset. {exc}")
        st.stop()

    city_profile, cluster_summary = build_climate_clusters(df)

    with st.sidebar:
        st.success(f"Loaded {len(df):,} rows from {df['country'].nunique()} countries.")
        countries = st.multiselect("Countries", sorted(df["country"].unique()))
        location_pool = df[df["country"].isin(countries)]["location_name"].unique() if countries else df["location_name"].unique()
        locations = st.multiselect("Locations", sorted(location_pool))
        seasons = st.multiselect("Seasons", sorted(df["season"].dropna().unique()))
        min_date = df["last_updated"].min().date()
        max_date = df["last_updated"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) != 2:
            st.warning("Select both a start and end date to continue.")
            st.stop()

    filtered_df = apply_filters(df, countries, locations, seasons, date_range)
    if filtered_df.empty:
        st.warning("The current filters returned no rows. Try widening the date range or selecting fewer filters.")
        st.stop()

    filtered_profiles = city_profile[city_profile["location_name"].isin(filtered_df["location_name"].unique())].copy()
    if filtered_profiles.empty:
        filtered_profiles = city_profile.copy()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Geo Intelligence", "Climate Lab", "Compare Cities", "Forecast & Explorer"]
    )
    with tab1:
        overview_tab(filtered_df, filtered_profiles)
    with tab2:
        geo_tab(filtered_df)
    with tab3:
        climate_lab_tab(filtered_df, filtered_profiles, cluster_summary)
    with tab4:
        comparison_tab(filtered_df, filtered_profiles)
    with tab5:
        explorer_tab(filtered_df)


if __name__ == "__main__":
    main()
