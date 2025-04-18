import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import os

# --------------------
# Page Configuration
# --------------------
st.set_page_config(page_title="🌍 Disaster Risk Classifier", layout="wide")

st.title("🌍 Disaster-Prone Area Classification App")
st.markdown("Predict the **risk level** of a region based on disaster and demographic data.")

# --------------------
# Load Model Only
# --------------------
@st.cache_resource
def load_model():
    model_path = "disaster_model.pkl"
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    return model

model = load_model()

if model is None:
    st.error("❌ Model file not found. Please ensure 'disaster_model.pkl' is available in the app directory.")
    st.stop()

# --------------------
# User Input Form
# --------------------
st.header("📥 Enter Region Data")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        disaster_score = st.slider("Disaster History Score (0-10)", 0, 10, 5)
        pop_density = st.number_input("Population Density (people/km²)", min_value=0, max_value=20000, value=5000, step=100)
        latitude = st.number_input("Latitude", min_value=5.0, max_value=40.0, value=22.5, step=0.1)
    with col2:
        urban_level = st.slider("Urbanization Level (0.0-1.0)", 0.0, 1.0, 0.5, step=0.05)
        houses_affected = st.number_input("Houses Affected", min_value=0, max_value=10000, value=100, step=10)
        deaths = st.number_input("Human Deaths", min_value=0, max_value=1000, value=10, step=1)
        longitude = st.number_input("Longitude", min_value=65.0, max_value=100.0, value=78.9, step=0.1)

    submitted = st.form_submit_button("🔍 Predict Risk")

# --------------------
# Prediction Logic
# --------------------
if submitted:
    # Feature Engineering
    risk_index = disaster_score * 0.5 + pop_density * 0.3 + urban_level * 0.2
    damage_scale = houses_affected + (deaths * 10)

    input_features = [[
        disaster_score, pop_density, urban_level,
        houses_affected, deaths, risk_index, damage_scale
    ]]

    try:
        prediction = model.predict(input_features)[0]  # model outputs string labels
        st.success(f"✅ **Predicted Risk Level: `{prediction}`**")

        # --------------------
        # Map Visualization
        # --------------------
        st.header("🗺️ Visualize Location on Risk Map")

        # Dark themed background
        map_object = folium.Map(
            location=[latitude, longitude],
            zoom_start=6,
            tiles="CartoDB dark_matter"
        )

        # User's location marker
        folium.Marker(
            location=[latitude, longitude],
            popup=folium.Popup(f"<b>Risk Level:</b> {prediction}<br><b>Damage Scale:</b> {damage_scale}", max_width=250),
            icon=folium.Icon(
                color=("red" if prediction == "High" else "orange" if prediction == "Medium" else "green"),
                icon="info-sign"
            )
        ).add_to(map_object)

        # Highlight high-risk states (static coordinates)
        high_risk_states = {
            "Bihar": [25.9, 85.1],
            "Assam": [26.2, 91.7],
            "Odisha": [20.9, 85.1],
            "Uttarakhand": [30.1, 79.3],
            "Tamil Nadu": [11.1, 78.7]
        }

        for state, coords in high_risk_states.items():
            folium.Circle(
                location=coords,
                radius=50000,  # 50 km radius
                color="red",
                fill=True,
                fill_opacity=0.3,
                popup=f"High Risk Area: {state}"
            ).add_to(map_object)

        # Save map to file (optional)
        map_file = "disaster_risk_map.html"
        map_object.save(map_file)

        # Display map inside Streamlit
        folium_static(map_object)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

    # --------------------
    # Optional: Load Full Risk Map
    # --------------------
    st.subheader("📍 View Full Risk Map")

    if os.path.exists("disaster_risk_map.html"):
        with open("disaster_risk_map.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    else:
        st.warning("⚠️ Full map file 'disaster_risk_map.html' not found.")
