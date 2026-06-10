from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #374151;
}

.status-card {
    background: #0f172a;
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #ef4444;
    margin-top: 10px;
}

.small-label {
    color: #9ca3af;
    font-size: 0.85rem;
}

.big-number {
    color: white;
    font-size: 2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# US 19 Interactive Safety Digital Twin

**Real-time scenario testing for roadway safety interventions between SR 60 and SR 686**
""")

with st.expander("How this digital twin works"):
    st.write("""
    The corridor is segmented into 500-foot roadway units. Each segment is assigned crash history,
    roadway characteristics, and safety indicators. A logistic regression model estimates predicted
    high-risk probability. Intervention packages modify roadway assumptions, and the model recalculates
    predicted risk in real time.
    """)

# -----------------------------
# LOAD DATA
# -----------------------------
gdf = gpd.read_file("us19_segments_final.geojson")
numeric_fields = ["SpeedLimit", "NumLanes", "risk_map", "high_risk"]

for field in numeric_fields:
    gdf[field] = pd.to_numeric(gdf[field], errors="coerce")
# Convert geometry to WGS84 for web mapping
if gdf.crs is not None:
    gdf = gdf.to_crs(epsg=4326)
 
gdf = gdf.dropna(subset=["SpeedLimit", "NumLanes", "risk_map", "high_risk"])
# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.markdown("### Preset Scenarios")

speed_reduction = 10
lane_reduction = 1
hotspot_only = False

preset = st.sidebar.selectbox(
    "Choose Intervention Package",
    [
        "Custom",
        "Speed Management",
        "Road Diet",
        "Access Management",
        "Signal Safety Upgrade",
        "Pedestrian Priority Package",
        "Aggressive Safety Redesign"
    ]
)

if preset == "Speed Management":
    speed_reduction = 10
    lane_reduction = 0
    hotspot_only = False
    package_cost = 250000

elif preset == "Road Diet":
    speed_reduction = 10
    lane_reduction = 2
    hotspot_only = False
    package_cost = 1200000

elif preset == "Access Management":
    speed_reduction = 5
    lane_reduction = 0
    hotspot_only = True
    package_cost = 600000

elif preset == "Signal Safety Upgrade":
    speed_reduction = 5
    lane_reduction = 0
    hotspot_only = True
    package_cost = 750000

elif preset == "Pedestrian Priority Package":
    speed_reduction = 10
    lane_reduction = 1
    hotspot_only = True
    package_cost = 900000

elif preset == "Aggressive Safety Redesign":
    speed_reduction = 20
    lane_reduction = 2
    hotspot_only = False
    package_cost = 2000000

else:
    package_cost = 0

st.sidebar.header("Scenario Controls")

speed_reduction = st.sidebar.slider(
    "Reduce Posted Speed Limit (mph)",
    0,
    20,
    speed_reduction
)

lane_reduction = st.sidebar.slider(
    "Lane Reduction Scenario",
    0,
    2,
    lane_reduction
)

# -----------------------------
# SCENARIO LOGIC
# -----------------------------
hotspot_only = st.sidebar.checkbox(
    "Apply treatment only to high-risk segments",
    value=False,
    key="hotspot_checkbox"
)

gdf["is_hotspot"] = gdf["risk_map"] >= 0.70
# Model feature engineering
gdf["intersection_hotspot"] = (gdf["risk_map"] >= 0.70).astype(int)

if "VulUser" in gdf.columns:
    gdf["vulnerable_user_issue"] = (gdf["VulUser"] > 0).astype(int)
else:
    gdf["vulnerable_user_issue"] = 0

if "NightTime" in gdf.columns:
    gdf["night_issue"] = (gdf["NightTime"] > 0).astype(int)
else:
    gdf["night_issue"] = 0

if "TotSerInj" in gdf.columns:
    gdf["severe_issue"] = (gdf["TotSerInj"] > 0).astype(int)
else:
    gdf["severe_issue"] = 0

if "LaneDepart" in gdf.columns:
    gdf["behavior_issue"] = (gdf["LaneDepart"] > 0).astype(int)
else:
    gdf["behavior_issue"] = 0

features = [
    "NumLanes",
    "SpeedLimit",
    "intersection_hotspot",
    "vulnerable_user_issue",
    "night_issue",
    "severe_issue",
    "behavior_issue"
]

X = gdf[features]
y = gdf["high_risk"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X, y)
gdf["model_baseline_risk"] = model.predict_proba(X)[:, 1]

if hotspot_only:
    treatment_mask = gdf["is_hotspot"]
else:
    treatment_mask = gdf["model_baseline_risk"] >= 0

gdf["scenario_speed"] = gdf["SpeedLimit"]
gdf["scenario_lanes"] = gdf["NumLanes"]

gdf.loc[treatment_mask, "scenario_speed"] = (
    gdf.loc[treatment_mask, "SpeedLimit"] - speed_reduction
)

gdf.loc[treatment_mask, "scenario_lanes"] = (
    gdf.loc[treatment_mask, "NumLanes"] - lane_reduction
)

gdf["scenario_lanes"] = gdf["scenario_lanes"].clip(lower=2)
gdf["scenario_speed"] = gdf["scenario_speed"].clip(lower=25)

scenario_X = gdf[features].copy()

scenario_X["SpeedLimit"] = gdf["SpeedLimit"]
scenario_X["NumLanes"] = gdf["NumLanes"]

scenario_X.loc[treatment_mask, "SpeedLimit"] = gdf.loc[treatment_mask, "scenario_speed"]
scenario_X.loc[treatment_mask, "NumLanes"] = gdf.loc[treatment_mask, "scenario_lanes"]

gdf["scenario_risk"] = model.predict_proba(scenario_X)[:, 1]
gdf["risk_change"] = gdf["scenario_risk"] - gdf["model_baseline_risk"]

gdf["recommended_strategy"] = "Monitoring"

gdf.loc[
    (gdf["model_baseline_risk"] >= 0.70),
    "recommended_strategy"
] = "Aggressive Safety Redesign"

gdf.loc[
    (gdf["model_baseline_risk"] >= 0.50) &
    (gdf["model_baseline_risk"] < 0.70),
    "recommended_strategy"
] = "Targeted Speed + Access Management"

# -----------------------------
# EXTRUSION HEIGHT
# -----------------------------
gdf["elevation"] = gdf["scenario_risk"] * 1000

# -----------------------------
# COLOR FUNCTION
# -----------------------------
def get_color(risk):
    if risk > 0.75:
        return [255, 0, 0, 210]
    elif risk > 0.5:
        return [255, 140, 0, 180]
    elif risk > 0.25:
        return [255, 255, 0, 170]
    else:
        return [0, 255, 120, 150]

gdf["display_risk"] = gdf["scenario_risk"]

gdf["color"] = gdf["display_risk"].apply(get_color)

points = gdf.copy()
points["geometry"] = points.geometry.centroid
points["lon"] = points.geometry.x
points["lat"] = points.geometry.y
points["display_risk"] = gdf["display_risk"]
points["height"] = points["display_risk"] * 700

hotspots = points[points["display_risk"] >= 0.50].copy()

hotspots["label"] = hotspots.apply(
    lambda row: f"High Risk | {row['recommended_strategy']}",
    axis=1
)

column_layer = pdk.Layer(
    "ColumnLayer",
    points,
    get_position=["lon", "lat"],
    get_elevation="height",
    elevation_scale=1.5,
    radius=35,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

hotspot_layer = pdk.Layer(
    "ScatterplotLayer",
    hotspots,
    get_position=["lon", "lat"],
    get_radius=85,
    get_fill_color=[255, 255, 255, 230],
    get_line_color=[255, 0, 0, 255],
    line_width_min_pixels=3,
    stroked=True,
    filled=True,
    pickable=True,
)
# -----------------------------
# PYDECK LAYER
# -----------------------------
layer = pdk.Layer(
    "GeoJsonLayer",
    gdf,
    opacity=1,
    stroked=True,
    filled=False,
    extruded=False,
    get_line_color="color",
    get_line_width=28,
    line_width_min_pixels=6,
    pickable=True,
)

# -----------------------------
# VIEW STATE
# -----------------------------
view_state = pdk.ViewState(
    latitude=27.915,
    longitude=-82.73,
    zoom=11.8,
    pitch=55,
    bearing=0
)

# -----------------------------
# RENDER MAP
# -----------------------------

tooltip = {
    "html": """
    <b>Predicted Risk:</b> {display_risk}<br/>
    <b>Recommended Strategy:</b> {recommended_strategy}<br/>
    <b>Speed:</b> {scenario_speed} mph<br/>
    <b>Lanes:</b> {scenario_lanes}
    """,
    "style": {
        "backgroundColor": "#111827",
        "color": "white",
        "border": "1px solid #374151",
        "borderRadius": "10px"
    }
}

view_state = pdk.ViewState(
    latitude=27.94,
    longitude=-82.73,
    zoom=11.2,
    pitch=50,
    bearing=10
)

# Baseline display data
baseline_gdf = gdf.copy()
baseline_gdf["display_risk"] = baseline_gdf["model_baseline_risk"]
baseline_gdf["color"] = baseline_gdf["display_risk"].apply(get_color)

baseline_points = baseline_gdf.copy()
baseline_points["geometry"] = baseline_points.geometry.centroid
baseline_points["lon"] = baseline_points.geometry.x
baseline_points["lat"] = baseline_points.geometry.y
baseline_points["height"] = baseline_points["display_risk"] * 2500

# Scenario display data
scenario_gdf = gdf.copy()
scenario_gdf["display_risk"] = scenario_gdf["scenario_risk"]
scenario_gdf["color"] = scenario_gdf["display_risk"].apply(get_color)

scenario_points = scenario_gdf.copy()
scenario_points["geometry"] = scenario_points.geometry.centroid
scenario_points["lon"] = scenario_points.geometry.x
scenario_points["lat"] = scenario_points.geometry.y
scenario_points["height"] = scenario_points["display_risk"] * 2500

hotspots = scenario_points[scenario_points["display_risk"] >= 0.50].copy()

# Layers
baseline_line_layer = pdk.Layer(
    "GeoJsonLayer",
    baseline_gdf,
    opacity=1,
    stroked=True,
    filled=False,
    extruded=False,
    get_line_color="color",
    get_line_width=26,
    line_width_min_pixels=5,
    pickable=True,
)

scenario_line_layer = pdk.Layer(
    "GeoJsonLayer",
    scenario_gdf,
    opacity=1,
    stroked=True,
    filled=False,
    extruded=False,
    get_line_color="color",
    get_line_width=26,
    line_width_min_pixels=5,
    pickable=True,
)

baseline_column_layer = pdk.Layer(
    "ColumnLayer",
    baseline_points,
    get_position=["lon", "lat"],
    get_elevation="height",
    elevation_scale=1.5,
    radius=45,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

scenario_column_layer = pdk.Layer(
    "ColumnLayer",
    scenario_points,
    get_position=["lon", "lat"],
    get_elevation="height",
    elevation_scale=1.5,
    radius=45,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

hotspot_layer = pdk.Layer(
    "ScatterplotLayer",
    hotspots,
    get_position=["lon", "lat"],
    get_radius=75,
    get_fill_color=[255, 255, 255, 210],
    get_line_color=[255, 0, 0, 230],
    line_width_min_pixels=2,
    stroked=True,
    filled=True,
    pickable=True,
)

# Decks
baseline_deck = pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=view_state,
    layers=[baseline_line_layer, baseline_column_layer],
    tooltip=tooltip,
)

scenario_deck = pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=view_state,
    layers=[scenario_line_layer, scenario_column_layer, hotspot_layer],
    tooltip=tooltip,
)

# Layout
map_left, map_right = st.columns(2)

with map_left:
    st.markdown("### Existing Conditions")
    st.pydeck_chart(baseline_deck, use_container_width=True, height=620)

with map_right:
    st.markdown("### Proposed Scenario")
    st.pydeck_chart(scenario_deck, use_container_width=True, height=620)

st.markdown("""
**Risk Legend:** 🟢 Low &nbsp;&nbsp; 🟡 Moderate &nbsp;&nbsp; 🟠 High &nbsp;&nbsp; 🔴 Severe  
**3D Tower Height:** taller towers = higher predicted crash risk
""")

output = gdf.drop(columns="geometry").to_csv(index=False)

st.download_button(
    label="Download Scenario Results",
    data=output,
    file_name="us19_scenario_results.csv",
    mime="text/csv"
)

# -----------------------------
# RESULTS
# -----------------------------

baseline_avg = gdf["model_baseline_risk"].mean()
scenario_avg = gdf["scenario_risk"].mean()
reduction_pct = ((baseline_avg - scenario_avg) / baseline_avg) * 100

baseline_hotspots = int((gdf["model_baseline_risk"] >= 0.70).sum())
scenario_hotspots = int((gdf["scenario_risk"] >= 0.70).sum())

if package_cost > 0:
    benefit_cost_score = reduction_pct / (package_cost / 1000000)
else:
    benefit_cost_score = 0

if preset == "Custom":
    recommendation = "Custom scenario selected. Adjust the controls to test different roadway safety assumptions."
elif reduction_pct >= 25:
    recommendation = f"{preset} produces a strong corridor-wide safety improvement and is recommended for priority implementation."
elif reduction_pct >= 10:
    recommendation = f"{preset} produces a moderate safety improvement and may be most effective when targeted to high-risk segments."
else:
    recommendation = f"{preset} produces limited corridor-wide risk reduction. Consider a stronger package or targeted hotspot treatment."

st.markdown("## Scenario Results")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Baseline Risk", round(baseline_avg, 2))
m2.metric("Scenario Risk", round(scenario_avg, 2))
m3.metric("Risk Reduction", f"{round(reduction_pct, 1)}%")
m4.metric("High-Risk Segments", f"{scenario_hotspots} / {baseline_hotspots}")

if package_cost > 0:
    m5.metric("Package Cost", f"${package_cost:,.0f}")
else:
    m5.metric("Package Cost", "Custom")

if package_cost > 0:
    st.metric("Risk Reduction per $1M", round(benefit_cost_score, 1))

st.markdown("## Planning Recommendation")

st.markdown(f"""
<div class="status-card">
{recommendation}
</div>
""", unsafe_allow_html=True)

comparison = pd.DataFrame({
    "Metric": [
        "Average Risk",
        "Highest Risk",
        "High-Risk Segments"
    ],
    "Existing Conditions": [
        round(gdf["model_baseline_risk"].mean(), 2),
        round(gdf["model_baseline_risk"].max(), 2),
        int((gdf["model_baseline_risk"] >= 0.70).sum())
    ],
    "Proposed Scenario": [
        round(gdf["scenario_risk"].mean(), 2),
        round(gdf["scenario_risk"].max(), 2),
        int((gdf["scenario_risk"] >= 0.70).sum())
    ]
})

st.markdown("## Scenario Comparison")
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.caption("Prototype developed for corridor-level safety scenario planning. Model outputs are exploratory and intended for planning support, not final engineering design.")