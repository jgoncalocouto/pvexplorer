
# pv_poa_app.py
# Streamlit app to estimate plane-of-array (POA) irradiance using pvlib clearsky models
# Run with: streamlit run pv_poa_app.py

from typing import Optional, List, Tuple

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


def integrate_power_kw_to_daily_energy(power_kw: pd.Series) -> pd.Series:
    """Integrate a kW power time-series to daily kWh using forward differences."""

    if power_kw.empty:
        return pd.Series(dtype=float)

    idx = power_kw.index
    values = power_kw.to_numpy(dtype=float)

    if len(values) == 1:
        delta_hours = np.zeros_like(values)
    else:
        diffs = np.diff(idx.asi8) / NS_PER_HOUR
        positive_diffs = diffs[diffs > 0]
        fallback = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
        diffs = np.where(diffs > 0, diffs, fallback)
        delta_hours = np.append(diffs, fallback)

    energy = values * delta_hours
    daily = pd.Series(energy, index=idx).resample("D").sum(min_count=1)
    daily.index.name = "date"
    return daily


def align_timestamp_to_index(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return *ts* converted to the timezone of *index* (if any)."""

    timestamp = pd.Timestamp(ts)
    idx_tz = index.tz if isinstance(index, pd.DatetimeIndex) else None

    if idx_tz is None:
        if timestamp.tzinfo is not None:
            return timestamp.tz_convert(None)
        return timestamp

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(idx_tz)
    return timestamp.tz_convert(idx_tz)


def select_representative_days(daily_series: pd.Series) -> List[Tuple[str, pd.Timestamp]]:
    """Pick days representing distribution extremes and spread for plotting."""

    if daily_series.empty:
        return []

    selections: List[Tuple[str, pd.Timestamp]] = []

    try:
        best_day = daily_series.idxmax()
        selections.append(("Best day", pd.Timestamp(best_day)))
    except ValueError:
        pass

    try:
        worst_day = daily_series.idxmin()
        selections.append(("Worst day", pd.Timestamp(worst_day)))
    except ValueError:
        pass

    median_value = daily_series.median()
    if not pd.isna(median_value):
        median_day = daily_series.sub(median_value).abs().idxmin()
        selections.append(("Median day", pd.Timestamp(median_day)))

        sigma = daily_series.std()
        if not pd.isna(sigma) and sigma > 0:
            low_target = median_value - 2 * sigma
            high_target = median_value + 2 * sigma
            low_day = daily_series.sub(low_target).abs().idxmin()
            high_day = daily_series.sub(high_target).abs().idxmin()
            selections.append(("Median − 2σ day", pd.Timestamp(low_day)))
            selections.append(("Median + 2σ day", pd.Timestamp(high_day)))

    # Filter out NaT entries that may slip through idx* lookups
    filtered = [(label, day) for label, day in selections if not pd.isna(day)]
    return filtered


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe (with index) to UTF-8 encoded CSV bytes for download."""

    return df.to_csv(index=True).encode("utf-8")


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

    st.header("PV System")
    dc_capacity_kwp = st.number_input("Array DC capacity (kWp)", min_value=0.1, value=4.64, step=0.1)
    global_efficiency = st.slider("Global DC efficiency", 0.0, 1.0, 0.85, step=0.01)
    inverter_capacity_kw = st.number_input("Inverter AC rating (kW)", min_value=0.1, value=4.0, step=0.1)

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


# Simple PV performance model (global efficiency + inverter clipping)
dc_kw = df["POA_Global"] / 1000.0 * dc_capacity_kwp * global_efficiency
ac_kw = dc_kw.clip(upper=inverter_capacity_kw)
df["DC_kW"] = dc_kw
df["AC_kW"] = ac_kw


# Daily totals in kWh/m^2/day for POA
daily_poa_kwh = compute_daily_energy_kwh(df["POA_Global"])
poa_summary = pd.DataFrame({
    "POA_kWh_per_m2": daily_poa_kwh,
})
monthly_summary = poa_summary.resample("MS").sum()

# Daily energy for DC/AC outputs
daily_dc_energy = integrate_power_kw_to_daily_energy(df["DC_kW"])
daily_ac_energy = integrate_power_kw_to_daily_energy(df["AC_kW"])
pv_summary = pd.DataFrame({
    "DC_Energy_kWh": daily_dc_energy,
    "AC_Energy_kWh": daily_ac_energy,
    "Clipping_Loss_kWh": daily_dc_energy - daily_ac_energy,
})
monthly_pv_summary = pv_summary.resample("MS").sum()

# KPI cards
col1, col2, col3 = st.columns(3)
total_kwh_m2 = daily_poa_kwh.sum()
peak_poa = df["POA_Global"].max()
day_with_max = daily_poa_kwh.idxmax() if not daily_poa_kwh.empty else None
col1.metric("Total POA (kWh/m²)", f"{total_kwh_m2:.1f}")
col2.metric("Peak POA (W/m²)", f"{peak_poa:.0f}")
col3.metric("Best day", f"{day_with_max.date() if day_with_max is not None else '—'}")

col4, col5, col6 = st.columns(3)
total_ac_energy = daily_ac_energy.sum()
total_dc_energy = daily_dc_energy.sum()
clipping_fraction = ((total_dc_energy - total_ac_energy) / total_dc_energy * 100.0) if total_dc_energy > 0 else 0.0
col4.metric("Total AC energy (kWh)", f"{total_ac_energy:.1f}")
col5.metric("Total DC energy (kWh)", f"{total_dc_energy:.1f}")
col6.metric("Clipping loss (%)", f"{clipping_fraction:.1f}")


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
    st.download_button(
        "Download POA time series (CSV)",
        data=dataframe_to_csv_bytes(df),
        file_name="poa_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
    )


with st.expander("Time series — PV power", expanded=True):
    pv_fig = go.Figure()
    pv_fig.add_trace(go.Scatter(x=df.index, y=df["DC_kW"], name="DC Power", mode="lines"))
    pv_fig.add_trace(go.Scatter(x=df.index, y=df["AC_kW"], name="AC Power (clipped)", mode="lines"))
    pv_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(pv_fig, use_container_width=True)
    st.download_button(
        "Download PV power time series (CSV)",
        data=dataframe_to_csv_bytes(df[["DC_kW", "AC_kW"]]),
        file_name="pv_power_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
        key="pv_power_download",
    )


with st.expander("Daily profile — AC power", expanded=False):
    representative_days = select_representative_days(daily_ac_energy)
    if not representative_days:
        st.info("Not enough daily AC energy data to build representative profiles.")
    else:
        profile_fig = go.Figure()
        for label, day in representative_days:
            aligned_day = align_timestamp_to_index(day, df.index)
            mask = df.index.normalize() == aligned_day.normalize()
            day_ac = df.loc[mask, "AC_kW"]
            if day_ac.empty:
                continue

            hours = ((day_ac.index - day_ac.index.normalize()) / pd.Timedelta(hours=1)).to_numpy()
            profile_fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=day_ac.values,
                    mode="lines",
                    name=f"{label} ({aligned_day.date()})",
                )
            )

        if not profile_fig.data:
            st.info("Unable to plot daily AC power profiles for the current dataset.")
        else:
            profile_fig.update_layout(
                xaxis_title="Hour of day",
                yaxis_title="AC power (kW)",
                hovermode="x unified",
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            profile_fig.update_xaxes(range=[0, 24])
            st.plotly_chart(profile_fig, use_container_width=True)


with st.expander("Daily energy tables", expanded=False):
    st.subheader("POA irradiance")
    st.dataframe(poa_summary)
    st.download_button(
        "Download daily POA energy (CSV)",
        data=dataframe_to_csv_bytes(poa_summary),
        file_name="poa_daily_energy.csv",
        mime="text/csv",
        use_container_width=True,
        key="poa_daily_download",
    )
    st.subheader("PV output")
    st.dataframe(pv_summary)
    st.download_button(
        "Download daily PV energy (CSV)",
        data=dataframe_to_csv_bytes(pv_summary),
        file_name="pv_daily_energy.csv",
        mime="text/csv",
        use_container_width=True,
        key="pv_daily_download",
    )
    st.subheader("Monthly POA energy")
    st.dataframe(monthly_summary.rename(columns={"POA_kWh_per_m2": "POA kWh/m²"}))
    st.download_button(
        "Download monthly POA energy (CSV)",
        data=dataframe_to_csv_bytes(monthly_summary),
        file_name="poa_monthly_energy.csv",
        mime="text/csv",
        use_container_width=True,
        key="poa_monthly_download",
    )
    st.subheader("Monthly PV energy")
    st.dataframe(monthly_pv_summary)
    st.download_button(
        "Download monthly PV energy (CSV)",
        data=dataframe_to_csv_bytes(monthly_pv_summary),
        file_name="pv_monthly_energy.csv",
        mime="text/csv",
        use_container_width=True,
        key="pv_monthly_download",
    )

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
    st.download_button(
        "Download daily POA totals (CSV)",
        data=dataframe_to_csv_bytes(poa_summary),
        file_name="poa_daily_totals.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Monthly totals table
with st.expander("Monthly totals (kWh/m² per month)", expanded=False):
    monthly_display = monthly_summary.rename(columns={"POA_kWh_per_m2": "POA kWh/m²"})
    st.dataframe(monthly_display)
    st.download_button(
        "Download monthly POA totals (CSV)",
        data=dataframe_to_csv_bytes(monthly_summary),
        file_name="poa_monthly_totals.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Model: pvlib Ineichen clear-sky or user-provided CSV irradiance (GHI/DNI/DHI). CSV timestamps are shifted to the selected timezone and present year.")
