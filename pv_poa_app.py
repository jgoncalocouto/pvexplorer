
# pv_poa_app.py
# Streamlit app to estimate plane-of-array (POA) irradiance using pvlib clearsky models
# Run with: streamlit run pv_poa_app.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pvlib.location import Location
import pvlib

st.set_page_config(page_title="PV POA Irradiance (pvlib)", layout="wide")

st.title("☀️ PV Plane-of-Array Irradiance — pvlib (clear-sky)")

with st.sidebar:
    st.header("Location & Time")
    lat = st.number_input("Latitude (°)", value=38.7223, format="%.6f")  # default Lisbon
    lon = st.number_input("Longitude (°)", value=-9.1393, format="%.6f")
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
    tilt = st.slider("Surface tilt (° from horizontal)", 0, 90, 25)
    azimuth = st.slider("Surface azimuth (°; 180 = South, 0/360 = North, 90 = East, 270 = West)", 0, 360, 180)
    albedo = st.slider("Ground albedo", 0.0, 0.9, 0.2, step=0.01)

    st.header("Sampling")
    freq_label = st.selectbox("Time step", ["15 min", "30 min", "1H"], index=2)
    freq = {"15 min": "15min", "30 min": "30min", "1H": "1H"}[freq_label]

st.markdown(
    """
This tool computes **clear-sky** irradiance using pvlib's Ineichen model and transposes it to the **plane-of-array (POA)** for your
specified tilt and azimuth. It's a great first step to size PV and understand seasonal/diurnal patterns. Later, you can switch to
measured or TMY weather to include clouds and real conditions.
""")

# Build time index
start = pd.Timestamp.combine(pd.to_datetime(date_start), pd.Timestamp.min.time()).tz_localize(tz)
end = pd.Timestamp.combine(pd.to_datetime(date_end), pd.Timestamp.max.time()).tz_localize(tz)

times = pd.date_range(start=start, end=end, freq=freq, tz=tz)

# Location object
site = Location(latitude=lat, longitude=lon, tz=tz, altitude=elevation, name="Site")

# Solar position
solpos = site.get_solarposition(times)

# Clear-sky (Ineichen)
clearsky = site.get_clearsky(times, model="ineichen")  # returns ghi, dni, dhi
# Standardize column names to upper-case for downstream access
clearsky = clearsky.rename(columns=str.upper)

# Airmass for temp model (optional - here just to demonstrate pipeline)
pressure = pvlib.atmosphere.alt2pres(elevation)
airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

# Transpose to plane of array
total_irr = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    dni=clearsky["DNI"],
    ghi=clearsky["GHI"],
    dhi=clearsky["DHI"],
    solar_zenith=solpos["apparent_zenith"],
    solar_azimuth=solpos["azimuth"],
    albedo=albedo,
)

# Combine results
df = pd.DataFrame({
    "GHI": clearsky["GHI"],
    "DNI": clearsky["DNI"],
    "DHI": clearsky["DHI"],
    "POA_Global": total_irr["poa_global"],
    "POA_Beam": total_irr["poa_direct"],
    "POA_Diffuse": total_irr["poa_sky_diffuse"],
    "POA_GroundReflected": total_irr["poa_ground_diffuse"],
})

# Daily totals in kWh/m^2/day for POA
poa_wh = df["POA_Global"].resample("D").sum(min_count=1) * (pd.to_timedelta(freq).total_seconds() / 3600.0) / 1000.0
poa_summary = pd.DataFrame({
    "POA_kWh_per_m2": poa_wh,
})
monthly_summary = poa_summary.resample("MS").sum()

# KPI cards
col1, col2, col3 = st.columns(3)
total_kwh_m2 = poa_wh.sum()
peak_poa = df["POA_Global"].max()
day_with_max = poa_wh.idxmax() if not poa_wh.empty else None
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
    bar.add_trace(go.Bar(x=poa_wh.index, y=poa_wh.values, name="Daily POA kWh/m²"))
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

st.caption("Model: Ineichen clear-sky via pvlib; no clouds or real-time weather. For measured/TMY weather, we can plug in EPW/TMY/NSRDB next.")
