import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("dermatology_random_forest.pkl", "rb"))
scaler = pickle.load(open("dermatology_scaler.pkl", "rb"))

# Feature names
features = [
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",
    "melanin_incontinence",
    "eosinophils_infiltrate",
    "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer",
    "spongiosis",
    "saw_tooth_appearance_of_retes",
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_monoluclear_infiltrate",
    "band_like_infiltrate",
    "age"
]

# Page
st.set_page_config(
    page_title="Dermatology AI",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 Dermatology Disease Classification")
st.write("Enter the patient feature values and predict the disease class.")

st.divider()

# Input fields
values = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    if i % 2 == 0:
        with col1:
            value = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                step=1.0
            )
            values.append(value)

    else:
        with col2:
            value = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                step=1.0
            )
            values.append(value)

st.divider()

# Prediction button
if st.button("🔍 Predict Disease", use_container_width=True):

    data = pd.DataFrame(
        [values],
        columns=features
    )

    # Scale input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)[0]

    st.success("Prediction Completed")

    st.subheader("🩺 Predicted Disease Class")

    st.info(str(prediction))