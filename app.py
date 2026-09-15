import streamlit as st

# ---------------------------------------------------------------------------
# Model: Multiple Linear Regression fitted via the Normal Equation
# Features (Set B - on-site sensors): IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE
# Target: AC_POWER
# Coefficients taken directly from the trained model in the notebook
# (Normal Equation was the best performer: R^2 = 0.9928, RMSE = 658.0)
# theta = [intercept, irradiation_coef, ambient_temp_coef, module_temp_coef]
# ---------------------------------------------------------------------------
THETA = [-588.761872, 27936.7160, 23.1577160, 6.76211514]
MODEL_RMSE = 658.02
MODEL_R2 = 0.9928


def predict_ac_power(irradiation: float, ambient_temp: float, module_temp: float) -> float:
    """Predict AC power (kW) from the three Set B weather values."""
    intercept, w_irr, w_amb, w_mod = THETA
    return (
        intercept
        + w_irr * irradiation
        + w_amb * ambient_temp
        + w_mod * module_temp
    )


st.set_page_config(page_title="AC Power Predictor", page_icon="\u2600\ufe0f", layout="centered")

st.title("\u2600\ufe0f Solar Plant AC Power Predictor")
st.caption(
    "Multiple linear regression (Normal Equation) trained on on-site weather "
    "sensor data \u2014 irradiation, ambient temperature, and module temperature."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=12, step=1)
    irradiation = st.number_input(
        "Irradiation (kW/m\u00b2)",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.01,
        format="%.3f",
        help="Typical range in the training data: 0.0 - 1.0 kW/m\u00b2",
    )

with col2:
    ambient_temp = st.number_input(
        "Ambient temperature (\u00b0C)",
        min_value=-10.0,
        max_value=60.0,
        value=28.0,
        step=0.5,
    )
    module_temp = st.number_input(
        "Module temperature (\u00b0C)",
        min_value=-10.0,
        max_value=80.0,
        value=35.0,
        step=0.5,
    )

st.divider()

# Sanity check: hour of day is not a model feature, but it's a useful
# context check against the physical plausibility of the inputs.
is_daylight_hour = 6 <= hour <= 18
if not is_daylight_hour and irradiation > 0.02:
    st.warning(
        f"You entered hour {hour}:00, which is outside typical daylight hours, "
        f"but irradiation is {irradiation:.3f} kW/m\u00b2. Double-check your inputs \u2014 "
        "predictions for implausible combinations may not be reliable."
    )

if st.button("Predict AC Power", type="primary", use_container_width=True):
    prediction = predict_ac_power(irradiation, ambient_temp, module_temp)
    prediction_clamped = max(prediction, 0.0)

    st.metric("Predicted AC Power", f"{prediction_clamped:,.1f} kW")

    if prediction < 0:
        st.caption(
            f"Raw model output was {prediction:,.1f} kW (negative, clamped to 0 above) "
            "\u2014 this usually happens for inputs far outside the training conditions "
            "(e.g. low irradiation / nighttime values)."
        )

    with st.expander("Model details"):
        st.write(f"**Method:** Normal Equation (closed-form linear regression)")
        st.write(f"**Test RMSE:** {MODEL_RMSE:.2f} kW")
        st.write(f"**Test R\u00b2:** {MODEL_R2:.4f}")
        st.write(
            "**Equation:** AC_POWER = "
            f"{THETA[0]:.2f} + {THETA[1]:.2f} \u00d7 Irradiation + "
            f"{THETA[2]:.2f} \u00d7 Ambient_Temp + {THETA[3]:.2f} \u00d7 Module_Temp"
        )
        st.caption(
            "Note: hour of day is not a feature in the trained model \u2014 it's used "
            "above only as a plausibility check on your inputs."
        )
else:
    st.info("Enter the weather values above and click **Predict AC Power**.")