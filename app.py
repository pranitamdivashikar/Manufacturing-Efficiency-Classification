import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Manufacturing Efficiency Dashboard", page_icon="🏭", layout="wide")

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv('Thales_Group_Manufacturing.csv')
    df.drop_duplicates(inplace=True)
    
    df['Sensor_Stability_Index'] = (df['Temperature_C'] / 100.0) * (df['Vibration_Hz'] + 1)
    df['Energy_Efficiency_Ratio'] = df['Production_Speed_units_per_hr'] / (df['Power_Consumption_kW'] + 1e-5)
    df['Error_Output_Ratio'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + 1e-5)
    df['Network_Reliability_Score'] = 100 - (df['Network_Latency_ms'] * 0.1 + df['Packet_Loss_%'] * 2)
    
    le_op = LabelEncoder()
    df['Operation_Mode_Encoded'] = le_op.fit_transform(df['Operation_Mode'])
    
    le_target = LabelEncoder()
    df['Efficiency_Status_Encoded'] = le_target.fit_transform(df['Efficiency_Status'])
    
    feature_cols = [
        'Machine_ID', 'Operation_Mode_Encoded', 'Temperature_C', 'Vibration_Hz',
        'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
        'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
        'Predictive_Maintenance_Score', 'Error_Rate_%',
        'Sensor_Stability_Index', 'Energy_Efficiency_Ratio',
        'Error_Output_Ratio', 'Network_Reliability_Score'
    ]
    
    X = df[feature_cols]
    y = df['Efficiency_Status_Encoded']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return df, model, le_op, le_target, feature_cols

df, model, le_op, le_target, feature_cols = load_data_and_model()

st.title("🏭 AI-Based Manufacturing Efficiency Classification Dashboard")

st.sidebar.header("🎛️ Factory Controls")
selected_machine = st.sidebar.selectbox("Select Machine ID", sorted(df['Machine_ID'].unique()))
selected_op_mode = st.sidebar.selectbox("Operation Mode", le_op.classes_)

temp = st.sidebar.slider("Temperature (°C)", float(df['Temperature_C'].min()), float(df['Temperature_C'].max()), float(df['Temperature_C'].mean()))
vibration = st.sidebar.slider("Vibration (Hz)", float(df['Vibration_Hz'].min()), float(df['Vibration_Hz'].max()), float(df['Vibration_Hz'].mean()))
power = st.sidebar.slider("Power Consumption (kW)", float(df['Power_Consumption_kW'].min()), float(df['Power_Consumption_kW'].max()), float(df['Power_Consumption_kW'].mean()))
latency = st.sidebar.slider("Network Latency (ms)", float(df['Network_Latency_ms'].min()), float(df['Network_Latency_ms'].max()), float(df['Network_Latency_ms'].mean()))
packet_loss = st.sidebar.slider("Packet Loss (%)", float(df['Packet_Loss_%'].min()), float(df['Packet_Loss_%'].max()), float(df['Packet_Loss_%'].mean()))
defect_rate = st.sidebar.slider("Defect Rate (%)", float(df['Quality_Control_Defect_Rate_%'].min()), float(df['Quality_Control_Defect_Rate_%'].max()), float(df['Quality_Control_Defect_Rate_%'].mean()))
prod_speed = st.sidebar.slider("Production Speed (units/hr)", float(df['Production_Speed_units_per_hr'].min()), float(df['Production_Speed_units_per_hr'].max()), float(df['Production_Speed_units_per_hr'].mean()))
maintenance_score = st.sidebar.slider("Predictive Maintenance Score", float(df['Predictive_Maintenance_Score'].min()), float(df['Predictive_Maintenance_Score'].max()), float(df['Predictive_Maintenance_Score'].mean()))
error_rate = st.sidebar.slider("Error Rate (%)", float(df['Error_Rate_%'].min()), float(df['Error_Rate_%'].max()), float(df['Error_Rate_%'].mean()))

op_encoded = le_op.transform([selected_op_mode])[0]
sensor_stability = (temp / 100.0) * (vibration + 1)
energy_eff = prod_speed / (power + 1e-5)
error_output = error_rate / (prod_speed + 1e-5)
network_reliability = 100 - (latency * 0.1 + packet_loss * 2)

input_data = pd.DataFrame([[
    selected_machine, op_encoded, temp, vibration, power, latency,
    packet_loss, defect_rate, prod_speed, maintenance_score, error_rate,
    sensor_stability, energy_eff, error_output, network_reliability
]], columns=feature_cols)

prediction_encoded = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]
predicted_label = le_target.inverse_transform([prediction_encoded])[0]
confidence = prediction_proba[prediction_encoded] * 100

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Efficiency Prediction")
    if predicted_label == 'High':
        st.success(f"**Predicted State:** {predicted_label}")
    elif predicted_label == 'Medium':
        st.warning(f"**Predicted State:** {predicted_label}")
    else:
        st.error(f"**Predicted State:** {predicted_label}")
    st.metric("Prediction Confidence", f"{confidence:.2f}%")

with col2:
    st.subheader("📊 Class Probabilities")
    proba_df = pd.DataFrame({'Class': le_target.classes_, 'Probability': prediction_proba})
    st.bar_chart(proba_df.set_index('Class'))
