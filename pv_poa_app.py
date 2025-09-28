
# pv_poa_app.py
# Streamlit app to estimate plane-of-array (POA) irradiance using pvlib clearsky models
# Run with: streamlit run pv_poa_app.py

from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pvlib.location import Location
import pvlib

NS_PER_HOUR = 3_600_000_000_000


def _safe_replace_year(ts: pd.Timestamp, target_year: int) -> Optional[pd.Timestamp]:
    """Return *ts* with the year replaced, handling leap years gracefully."""

    try:
        return ts.replace(year=target_year)
    except ValueError:
        if ts.month == 2 and ts.day == 29:
            return ts.replace(year=target_year, month=2, day=28)
    return None


def compute_daily_energy_kwh(series: pd.Series) -> pd.Series:
    """Integrate a W/m² time-series to daily kWh/m² using forward differences."""

    if series.empty:
        return pd.Series(dtype=float)

    idx = series.index
    values = series.to_numpy(dtype=float)

    if len(values) == 1:
        delta_hours = np.zeros_like(values)
    else:
        diffs = np.diff(idx.asi8) / NS_PER_HOUR
        positive_diffs = diffs[diffs > 0]
        fallback = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
        diffs = np.where(diffs > 0, diffs, fallback)
        delta_hours = np.append(diffs, fallback)

    energy = values * delta_hours / 1000.0
    daily = pd.Series(energy, index=idx).resample("D").sum(min_count=1)
    daily.index.name = "date"
    return daily


def process_csv_irradiance(
    raw_df: pd.DataFrame,
    timestamp_col: str,
    ghi_col: str,
    dni_col: str,
    dhi_col: str,
    tz: str,
    target_year: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return a cleaned, timezone-aware irradiance dataframe sourced from CSV."""

    working = raw_df[[timestamp_col, ghi_col, dni_col, dhi_col]].copy()
    working.columns = ["timestamp", "GHI", "DNI", "DHI"]

    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"])

    for col in ["GHI", "DNI", "DHI"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=["GHI", "DNI", "DHI"])

    if working.empty:
        return pd.DataFrame(columns=["GHI", "DNI", "DHI"])

    timestamps = pd.DatetimeIndex(working.pop("timestamp"))

    if timestamps.tz is None:
        localized = timestamps.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    else:
        localized = timestamps.tz_convert(tz)

    valid_mask = ~localized.isna()
    working = working.loc[valid_mask]
    localized = localized[valid_mask]

    aligned = localized.map(lambda ts: _safe_replace_year(ts, target_year))
    aligned = pd.Index(aligned)
    aligned_mask = aligned.notna()
    working = working.loc[aligned_mask]
    aligned = aligned[aligned_mask]

    if working.empty:
        return pd.DataFrame(columns=["GHI", "DNI", "DHI"])

    working.index = pd.DatetimeIndex(aligned).tz_convert(tz)
    working = working.sort_index()
    working = working.groupby(level=0).mean()

    window = working.loc[(working.index >= window_start) & (working.index <= window_end)]
    return window

def build_azimuth_compass(angle_deg: float) -> go.Figure:
    """Return a compass-style polar chart highlighting the chosen azimuth."""

    base_angles = np.linspace(0, 360, 361)
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=np.ones_like(base_angles),
            theta=base_angles,
            mode="lines",
            line=dict(color="#CCCCCC", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for direction in [0, 90, 180, 270]:
        fig.add_trace(
            go.Scatterpolar(
                r=[0, 1],
                theta=[direction, direction],
                mode="lines",
                line=dict(color="#BBBBBB", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatterpolar(
            r=[1.05, 1.05, 1.05, 1.05],
            theta=[0, 90, 180, 270],
            mode="text",
            text=["N", "E", "S", "W"],
            textfont=dict(size=16, color="#444444"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[0, 0.82],
            theta=[angle_deg, angle_deg],
            mode="lines",
            line=dict(color="#FF8C00", width=6),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[0.9],
            theta=[angle_deg],
            mode="markers",
            marker=dict(color="#FF8C00", size=16, symbol="triangle-up"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"Azimuth: {angle_deg}°",
        showlegend=False,
        polar=dict(
            angularaxis=dict(rotation=90, direction="clockwise", showticklabels=False, ticks=""),
            radialaxis=dict(range=[0, 1.1], showticklabels=False, ticks="", showgrid=False, visible=False),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=320,
    )

    return fig


st.set_page_config(page_title="PV POA Irradiance (pvlib)", layout="wide")

st.title("☀️ PV Plane-of-Array Irradiance — pvlib (clear-sky)")

csv_df: Optional[pd.DataFrame] = None
timestamp_col = ghi_col = dni_col = dhi_col = None

with st.sidebar:
    st.header("Location & Time")
    lat = st.number_input("Latitude (°)", value=41.4836, format="%.4f")  # default Lisbon
    lon = st.number_input("Longitude (°)", value=-8.55, format="%.2f")
    map_zoom = st.slider("Map zoom", min_value=1, max_value=15, value=10)
    location_point = pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]})
    st.map(location_point, zoom=map_zoom, use_container_width=True)
    tz = st.text_input("Timezone (IANA)", value="Europe/Lisbon")
    elevation = st.number_input("Elevation (m)", value=100, step=10)
    dates = st.date_input(
        "Date range",
        value=(pd.Timestamp.today(tz=tz).date().replace(month=1, day=1),
               pd.Timestamp.today(tz=tz).date().replace(month=12, day=31)),
    )
    if isinstance(dates, tuple):
        date_start, date_end = dates
    else:
        date_start = dates
        date_end = dates

    st.header("Array Geometry")
    tilt = st.slider("Surface tilt (° from horizontal)", 0, 90, 30)
    azimuth = st.slider(
        "Surface azimuth (°; 180 = South, 0/360 = North, 90 = East, 270 = West)",
        0,
        360,
        200,
    )

    st.subheader("Orientation preview")
    st.plotly_chart(build_azimuth_compass(azimuth), width="stretch")

    albedo = st.slider("Ground albedo", 0.0, 0.9, 0.6, step=0.01)

    st.header("Sampling")
    freq_label = st.selectbox("Time step", ["15 min", "30 min", "1h"], index=2)
    freq = {"15 min": "15min", "30 min": "30min", "1h": "1h"}[freq_label]

    st.header("Irradiance source")
    source_label = st.radio(
        "Use clear-sky or upload measured data?",
        (
            "Clear-sky (pvlib Ineichen)",
            "Upload CSV (map to GHI/DNI/DHI)",
        ),
    )

    if source_label == "Upload CSV (map to GHI/DNI/DHI)":
        uploaded_file = st.file_uploader("CSV with irradiance columns", type=["csv"])
        if uploaded_file is not None:
            csv_df = pd.read_csv(uploaded_file)
            st.caption(f"Loaded {csv_df.shape[0]} rows × {csv_df.shape[1]} columns")
            if not csv_df.empty:
                timestamp_col = st.selectbox(
                    "Timestamp column",
                    options=csv_df.columns.tolist(),
                    key="timestamp_column_select",
                )
                remaining = [c for c in csv_df.columns if c != timestamp_col]
                if len(remaining) < 3:
                    st.warning("Pick a CSV with columns for GHI, DNI, and DHI.")
                else:
                    ghi_col = st.selectbox(
                        "Column → GHI",
                        options=remaining,
                        key="ghi_column_select",
                    )
                    remaining_dni = [c for c in remaining if c != ghi_col]
                    dni_col = st.selectbox(
                        "Column → DNI",
                        options=remaining_dni,
                        key="dni_column_select",
                    )
                    remaining_dhi = [c for c in remaining_dni if c != dni_col]
                    if remaining_dhi:
                        dhi_col = st.selectbox(
                            "Column → DHI",
                            options=remaining_dhi,
                            key="dhi_column_select",
                        )
                    else:
                        st.warning("Select distinct columns for each irradiance component.")
            else:
                st.warning("Uploaded file is empty.")

st.markdown(
    """
This tool computes plane-of-array (POA) irradiance using either pvlib's **clear-sky Ineichen model** or **user-supplied CSV irradiance data**. Map your CSV columns to GHI/DNI/DHI to evaluate measured or TMY datasets, or stick with the clear-sky baseline for quick feasibility scans.
""")




target_year = pd.Timestamp.today(tz=tz).year
start = pd.Timestamp.combine(pd.to_datetime(date_start), pd.Timestamp.min.time()).tz_localize(tz)
end = pd.Timestamp.combine(pd.to_datetime(date_end), pd.Timestamp.max.time()).tz_localize(tz)

# Location object
site = Location(latitude=lat, longitude=lon, tz=tz, altitude=elevation, name="Site")

if source_label == "Clear-sky (pvlib Ineichen)":
    times = pd.date_range(start=start, end=end, freq=freq, tz=tz)
    clearsky = site.get_clearsky(times, model="ineichen").rename(columns=str.upper)
    irradiance_components = clearsky[["GHI", "DNI", "DHI"]]
else:
    if csv_df is None or not all([timestamp_col, ghi_col, dni_col, dhi_col]):
        st.info("Upload a CSV and map each irradiance component to continue.")
        st.stop()

    csv_processed = process_csv_irradiance(
        csv_df,
        timestamp_col,
        ghi_col,
        dni_col,
        dhi_col,
        tz,
        target_year,
        start,
        end,
    )

    if csv_processed.empty:
        st.error("No rows remain after cleaning. Adjust the date range, timezone, or CSV mappings.")
        st.stop()

    times = csv_processed.index
    irradiance_components = csv_processed

if source_label != "Clear-sky (pvlib Ineichen)":
    st.success(f"Loaded {irradiance_components.shape[0]} timestamps from the CSV after cleaning and year alignment.")
    with st.expander("CSV irradiance preview", expanded=False):
        st.dataframe(irradiance_components.head(100))

# Solar position
solpos = site.get_solarposition(times)


# Airmass for temp model (optional - here just to demonstrate pipeline)
pressure = pvlib.atmosphere.alt2pres(elevation)
airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

# Transpose to plane of array
total_irr = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=irradiance_components["DNI"],
    ghi=irradiance_components["GHI"],
    dhi=irradiance_components["DHI"],
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    albedo=albedo,
)


# Combine results
df = pd.DataFrame({
    "GHI": irradiance_components["GHI"],
    "DNI": irradiance_components["DNI"],
    "DHI": irradiance_components["DHI"],
    "POA_Global": total_irr["poa_global"],
    "POA_Beam": total_irr["poa_direct"],
    "POA_Diffuse": total_irr["poa_sky_diffuse"],
    "POA_GroundReflected": total_irr["poa_ground_diffuse"],
})


# Daily totals in kWh/m^2/day for POA
daily_poa_kwh = compute_daily_energy_kwh(df["POA_Global"])
poa_summary = pd.DataFrame({
    "POA_kWh_per_m2": daily_poa_kwh,
})
monthly_summary = poa_summary.resample("MS").sum()

# KPI cards
col1, col2, col3 = st.columns(3)
total_kwh_m2 = daily_poa_kwh.sum()
peak_poa = df["POA_Global"].max()
day_with_max = daily_poa_kwh.idxmax() if not daily_poa_kwh.empty else None
col1.metric("Total POA (kWh/m²)", f"{total_kwh_m2:.1f}")
col2.metric("Peak POA (W/m²)", f"{peak_poa:.0f}")
col3.metric("Best day", f"{day_with_max.date() if day_with_max is not None else '—'}")


# Plot time series (POA components)
with st.expander("Time series — POA components", expanded=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["POA_Global"], name="POA Global", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["POA_Beam"], name="POA Beam", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["POA_Diffuse"], name="POA Diffuse (sky)", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["POA_GroundReflected"], name="POA Ground-reflected", mode="lines"))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Irradiance (W/m²)",
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# Daily totals bar
with st.expander("Daily POA energy (kWh/m²/day)", expanded=False):
    bar = go.Figure()
    bar.add_trace(go.Bar(x=daily_poa_kwh.index, y=daily_poa_kwh.values, name="Daily POA kWh/m²"))
    bar.update_layout(
        xaxis_title="Day",
        yaxis_title="kWh/m²/day",
        hovermode="x",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)

# Monthly totals table
with st.expander("Monthly totals (kWh/m² per month)", expanded=False):
    st.dataframe(monthly_summary.rename(columns={"POA_kWh_per_m2": "POA kWh/m²"}))

st.caption("Model: pvlib Ineichen clear-sky or user-provided CSV irradiance (GHI/DNI/DHI). CSV timestamps are shifted to the selected timezone and present year.")
