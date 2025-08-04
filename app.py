import streamlit as st
import pandas as pd
import datetime
import altair as alt
from model import train
from sklearn.metrics import r2_score

# Load models and trained feature columns
@st.cache_data
def train_models():
    lr_model, rf_model, mse_lr, mse_rf, feature_columns = train()
    return lr_model, rf_model, mse_lr, mse_rf, feature_columns

def preprocess_input(df, feature_columns):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['season'] = df['date'].dt.month % 12 // 3 + 1

    features = ['T2', 'T6', 'RH_1', 'RH_2', 'T_out', 'RH_out', 'Windspeed',
                'hour', 'day', 'weekday', 'is_weekend', 'season']
    
    X = df[features]
    X = pd.get_dummies(X, columns=['season'], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X

# Page Config
st.set_page_config(page_title="🏠 Energy Predictor", layout="centered")
st.title("⚡ Home Energy Consumption Predictor")
st.markdown("Predict energy usage (Watts) using temperature, humidity, windspeed, and time-based features.")

# Load models
lr_model, rf_model, mse_lr, mse_rf, feature_columns = train_models()

# Sidebar for input
st.sidebar.header("🔧 Input Features")
date = st.sidebar.date_input("Date", datetime.date.today())
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
T2 = st.sidebar.number_input("Temperature T2 (°C)", value=20.0)
T6 = st.sidebar.number_input("Temperature T6 (°C)", value=21.0)
RH_1 = st.sidebar.number_input("Humidity RH_1 (%)", value=40.0)
RH_2 = st.sidebar.number_input("Humidity RH_2 (%)", value=45.0)
T_out = st.sidebar.number_input("Outside Temperature (°C)", value=15.0)
RH_out = st.sidebar.number_input("Outside Humidity (%)", value=50.0)
Windspeed = st.sidebar.number_input("Windspeed (m/s)", value=5.0)

# Create single input row
input_df = pd.DataFrame({
    'date': [datetime.datetime.combine(date, datetime.time(hour))],
    'T2': [T2],
    'T6': [T6],
    'RH_1': [RH_1],
    'RH_2': [RH_2],
    'T_out': [T_out],
    'RH_out': [RH_out],
    'Windspeed': [Windspeed]
})

# Process input
X_input = preprocess_input(input_df.copy(), feature_columns)

if st.button("🔍 Predict Now"):
    pred_lr = lr_model.predict(X_input)[0]
    pred_rf = rf_model.predict(X_input)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔹 Linear Regression Prediction", f"{pred_lr:.2f} Watts", delta=f"MSE: {mse_lr:.2f}")
        st.markdown(f"**R² Score:** `{r2_score([pred_rf], [pred_lr]):.3f}`")
    with col2:
        st.metric("🌲 Random Forest Prediction", f"{pred_rf:.2f} Watts", delta=f"MSE: {mse_rf:.2f}")
        st.markdown(f"**R² Score:** `{r2_score([pred_lr], [pred_rf]):.3f}`")

    st.success("✅ Prediction complete!")

    st.markdown("🔍 **Tips to Reduce Energy Usage**")
    st.markdown("""
    - Use smart thermostats to schedule usage  
    - Avoid peak-time usage  
    - Use natural light during the day  
    - Seal air leaks and insulate walls  
    """)

# Batch Prediction
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")
csv_file = st.file_uploader("Upload CSV file with same input columns", type=["csv"])

if csv_file is not None:
    df_batch = pd.read_csv(csv_file)
    
    try:
        X_batch = preprocess_input(df_batch.copy(), feature_columns)
        df_batch["Linear_Prediction_Watts"] = lr_model.predict(X_batch)
        df_batch["RandomForest_Prediction_Watts"] = rf_model.predict(X_batch)
        st.success("✅ Batch predictions complete!")

        st.dataframe(df_batch.head())

        # Visualization with Altair
        df_viz = df_batch.reset_index().rename(columns={"index": "Data Point"})

        chart = alt.Chart(df_viz).mark_line().encode(
            x=alt.X("Data Point", title="Data Point Index"),
            y=alt.Y("RandomForest_Prediction_Watts", title="Energy (Watts)"),
            tooltip=["Data Point", "RandomForest_Prediction_Watts"]
        ).properties(
            title="🔋 Random Forest: Home Energy Predictions",
            width=700,
            height=400
        ).configure_title(fontSize=20)

        st.altair_chart(chart, use_container_width=True)

        # Download Predictions
        csv_download = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

st.markdown("---")

