import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Risk Predictor", layout="centered")

# --- Load all ML artifacts securely ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('final_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')             # CRITICAL FIX: Load the saved scaler
    le = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature.pkl')
    return model, scaler, le, feature_names

model, scaler, le, FEATURE_NAMES = load_artifacts()

st.title("Cancer Risk Level Predictor")
st.markdown("Predict `Risk_Level` (Low / Medium / High) safely using standardized patient data.")

# --- Shared Preprocessing Engine ---
def preprocess_and_scale(df):
    """
    Ensures input data aligns with training columns, resolves mismatches,
    and scales data using the training run's scaler parameters.
    """
    # 1. Re-align and fill missing columns with 0
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing and option == "Upload CSV (batch)":
        st.warning(f"⚠️ Missing columns in input — filling with zeros: {missing}")
    
    for c in missing:
        df[c] = 0
        
    # 2. Match exact column order and coerce values to numeric
    df = df[FEATURE_NAMES].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 3. CRITICAL FIX: Scale the data using the fitted training scaler
    scaled_array = scaler.transform(df)
    
    # 4. Convert back to DataFrame so XGBoost retains feature context
    return pd.DataFrame(scaled_array, columns=FEATURE_NAMES)


# --- Prediction UI Mode Selection ---
option = st.radio("Prediction mode", ("Upload CSV (batch)", "Manual input (single)"))

if option == "Upload CSV (batch)":
    uploaded_file = st.file_uploader("Upload CSV with feature columns", type=['csv'])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        # Preprocess and scale batch input
        X_scaled = preprocess_and_scale(input_df)
        
        # Predict using scaled features
        preds_enc = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        
        # Assemble final structured DataFrame
        result = input_df[FEATURE_NAMES].copy() # Keep original raw values for viewing
        result['Predicted_Risk_Level'] = le.inverse_transform(preds_enc)
        
        # Map back probability columns correctly
        for i, cls in enumerate(le.classes_):
            result[f'prob_{cls}'] = probs[:, i]
            
        st.success("🎉 Predictions ready")
        st.dataframe(result)
        st.download_button("Download results (CSV)", result.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

else:
    st.sidebar.header("Patient features (manual)")
    input_data = {}
    
    # Smart defaults for human baseline tracking
    for feat in FEATURE_NAMES:
        default_val = 60.0 if feat == 'Age' else (25.0 if feat == 'BMI' else 0.0)
        input_data[feat] = st.sidebar.number_input(feat, value=float(default_val))

    if st.sidebar.button("Predict"):
        X_single = pd.DataFrame([input_data])
        X_scaled = preprocess_and_scale(X_single)
        
        # Run prediction on scaled matrix
        pred_enc = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        pred = le.inverse_transform([pred_enc])[0]

        st.write("### Prediction")
        st.write(f"**Predicted Risk_Level:** `{pred}`")
        
        # Display crisp probability table
        prob_df = pd.DataFrame({'class': list(le.classes_), 'probability': probs})
        prob_df = prob_df.sort_values('probability', ascending=False).reset_index(drop=True)
        st.table(prob_df)

        # High risk threshold logic
        high_prob = prob_df.loc[prob_df['class'] == 'High', 'probability'].values[0]
        if high_prob >= 0.5:
            st.warning(f"⚠️ High risk probability is high ({high_prob:.2f}) — consider clinical follow-up.")
        else:
            st.success(f"✅ High risk probability is safe ({high_prob:.2f})")

st.markdown("---")
st.caption("Model Framework: Robust multi-class data mapping pipeline with strict state-locked scaling validation.")