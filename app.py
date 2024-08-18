import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import numpy as np
from tensorflow.keras.models import load_model
from statsmodels.tsa.seasonal import seasonal_decompose

# Set page config
st.set_page_config(page_title="Weather Prediction App 🌦️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .info-box {
        background-color: #e9f7ef;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        max-width: 800px;
        margin: 0 auto;  /* Center the box */
    }
    .info-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px auto;
    }
    .info-table th, .info-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .info-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# File paths for models and data
MODEL_DIR = 'models'
DATA_DIR = 'data'
NN_MODEL_FILE_RAW = os.path.join(MODEL_DIR, 'neural_network_model_imputed.keras')
NN_MODEL_FILE_CLEAN = os.path.join(MODEL_DIR, 'neural_network_model_clean.keras')
SCALER_FILE_CLEAN = os.path.join(MODEL_DIR, 'scaler_clean.pkl')
SCALER_FILE_IMPUTED = os.path.join(MODEL_DIR, 'scaler_imputed.pkl')
SHAP_VALUES_FILE_CLEAN = 'models/shap_values_clean.pkl'  # SHAP values for clean data
SHAP_VALUES_FILE_RAW = 'models/shap_values_imputed.pkl'  # SHAP values for raw data

# Function to load data based on selected dataset type
@st.cache_data
def load_data(data_type):
    if data_type == "Clean":
        df = pd.read_parquet(os.path.join(DATA_DIR, 'processed/df_clean.parquet'))
        X_train = joblib.load(os.path.join(DATA_DIR, 'X_trained_clean.joblib'))
        X_test = joblib.load(os.path.join(DATA_DIR, 'X_test_clean.joblib'))
    else:  # Raw data with outliers
        df = pd.read_parquet(os.path.join(DATA_DIR, 'processed/df_imputed.parquet'))
        X_train = joblib.load(os.path.join(DATA_DIR, 'X_trained_imputed.joblib'))
        X_test = joblib.load(os.path.join(DATA_DIR, 'X_test_imputed.joblib'))
    return df, X_train, X_test

# Function to load Neural Network model based on dataset type
def load_neural_network_model(data_type):
    if data_type == "Clean":
        model_path = NN_MODEL_FILE_CLEAN
    else:
        model_path = NN_MODEL_FILE_RAW

    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success(f"✅ Neural Network model ({data_type}) loaded successfully")
        return model
    else:
        st.error(f"❌ Neural Network model file for {data_type} data not found")
        return None

# SHAP Explanation text
def shap_explanation():
    explanation = """
    **Understanding the SHAP Summary Plot:**

    - **Tdew (degC)**: Dew point temperature has a significant impact on the model’s predictions. High dew point temperatures (red dots) generally increase the predicted outcome, while low dew points (blue dots) tend to reduce it.

    - **rh (%)**: Relative humidity is another influential feature. Higher humidity (red dots) usually decreases the predicted outcome, showing a strong negative impact on predictions.

    - **wd (deg)**: Wind direction does not have a significant effect on the predictions. The SHAP values are concentrated around zero, indicating minimal influence.

    - **p (mbar)**: Pressure also has little impact on the predictions. Most SHAP values are close to zero, suggesting that this feature doesn't strongly influence the model.

    - **wv (m/s)**: Wind speed has a low impact on the predictions as well, with SHAP values clustered around zero.

    Overall, **Tdew (degC)** and **rh (%)** are the most important features, driving the model’s predictions. The other features contribute much less.
    """
    st.markdown(explanation, unsafe_allow_html=True)

# Main app
def main():
    st.title("☀️ Weather Prediction App 🌡️")

    # Sidebar - Select Dataset
    data_type = st.sidebar.selectbox("Choose Dataset Type", ["Clean", "Raw"])

    # Load data based on the selection
    df, X_train, X_test = load_data(data_type)

    # Convert X_test to NumPy array if it is a DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # Load the appropriate scaler based on dataset type
    if data_type == "Clean":
        scaler = joblib.load(SCALER_FILE_CLEAN)
    else:
        scaler = joblib.load(SCALER_FILE_IMPUTED)

    # Load the Neural Network model
    nn_model = load_neural_network_model(data_type)

    if nn_model is None:
        return  # Exit if model is not loaded

    # Sidebar - Navigation
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Go to", ["📌 Data Info", "📈 EDA", "🤖 Model & Prediction"])

    if page == "📌 Data Info":
        st.header("📊 Dataset Information")

        # Add "Understanding the Weather Data" information
        st.subheader("Understanding the Weather Data")
        st.markdown("""
        <div class="info-box">
        <table class="info-table">
            <tr>
                <th>Weather Stuff</th>
                <th>What It Means</th>
                <th>How It Helps Our Weather AI</th>
            </tr>
            <tr>
                <td><strong>Date Time</strong></td>
                <td>When the weather was measured. Like a timestamp for each weather snapshot.</td>
                <td>Very important! Helps the AI understand patterns over time, like daily or seasonal changes.</td>
            </tr>
            <tr>
                <td><strong>Temperature (T)</strong></td>
                <td>How hot or cold it is outside, measured in degrees Celsius.</td>
                <td>Super important! One of the main things we want to predict in weather forecasting.</td>
            </tr>
            <tr>
                <td><strong>Pressure (p)</strong></td>
                <td>The weight of the air pushing down on us. Measured in millibars.</td>
                <td>Very useful! Changes in pressure often mean changes in weather are coming.</td>
            </tr>
            <tr>
                <td><strong>Humidity (rh)</strong></td>
                <td>How much water vapor is in the air, shown as a percentage.</td>
                <td>Really helpful! High humidity can mean rain is likely, and it affects how hot it feels.</td>
            </tr>
            <tr>
                <td><strong>Wind Speed (wv)</strong></td>
                <td>How fast the wind is blowing, measured in meters per second.</td>
                <td>Important! Wind can bring weather changes and affects how temperature feels.</td>
            </tr>
            <tr>
                <td><strong>Wind Direction (wd)</strong></td>
                <td>Which way the wind is coming from, measured in degrees (like a compass).</td>
                <td>Useful! The wind direction can tell us about incoming weather patterns.</td>
            </tr>
            <tr>
                <td><strong>Dew Point (Tdew)</strong></td>
                <td>The temperature at which water vapor turns into dew. Close to real temperature means it might rain.</td>
                <td>Helpful! Good for predicting rain and how comfortable the weather will feel.</td>
            </tr>
            <tr>
                <td><strong>Rain</strong></td>
                <td>This dataset doesn't directly measure rain, but we can guess it from other measurements.</td>
                <td>We can teach our AI to predict rain using other variables like humidity and dew point!</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"Total records: {len(df)}")
        st.write("🔍 First few rows of the dataset:")
        st.dataframe(df.head())

        st.subheader("📝 Data Description")
        st.write(df.describe())

    elif page == "📈 EDA":
        st.header("🔍 Exploratory Data Analysis")

        # Ensure the 'Date Time' column is in datetime format
        df['Date Time'] = pd.to_datetime(df['Date Time'])

        # Temperature over time
        st.subheader("🌡️ Temperature Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date Time'], df['T (degC)'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Variation Over Time')
        st.pyplot(fig)

        # Seasonal Decomposition
        st.subheader("🔄 Seasonal Decomposition")
        daily_temp = df.set_index('Date Time')['T (degC)'].resample('D').mean()
        daily_temp = daily_temp.fillna(method='ffill')
        result = seasonal_decompose(daily_temp, model='additive', period=365)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        result.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')
        st.pyplot(fig)

        # Monthly Temperature Distribution
        st.subheader("📅 Monthly Temperature Distribution")
        df['Month'] = df['Date Time'].dt.month
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Month', y='T (degC)', data=df, ax=ax)
    
        ax.set_xlabel('Month')
        ax.set_ylabel('Temperature (°C)')
        st.pyplot(fig)

    else:  # Model & Prediction
        st.header("🤖 Model and Prediction")

        features = ['p (mbar)', 'rh (%)', 'wv (m/s)', 'wd (deg)', 'Tdew (degC)']

        # Interactive Prediction
        st.subheader("🔮 Make a Prediction")
        user_input = {}
        input_scaled = None  # Initialize input_scaled to None

        # Use correct indexing for sliders, assuming X_test is now a NumPy array
        for i, feature in enumerate(features):
            user_input[feature] = st.slider(
                f"Select {feature}",
                float(X_test[:, i].min()),  # Use column-wise min
                float(X_test[:, i].max()),  # Use column-wise max
                float(X_test[:, i].mean())  # Use column-wise mean
            )
        st.write(user_input)

        if st.button("🚀 Predict"):
            try:
                # Prepare the input for prediction
                input_values = np.array([[user_input[feature] for feature in features]])
                input_scaled = scaler.transform(input_values)

                # Neural Network Prediction
                nn_prediction = nn_model.predict(input_scaled)

                # Display the predicted temperature
                st.markdown(f"""
                    <div style="background-color: #17a2b8; color: white; padding: 20px; border-radius: 10px; font-size: 30px; text-align: center;">
                        🌡️ <strong>Predicted Temperature: {nn_prediction[0][0]:.2f}°C</strong>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.error("Please make sure all input values are within the expected range.")

        # SHAP values for Neural Network
        st.subheader("🔑 SHAP Values (Neural Network)")
        
        try:
            # Load precomputed SHAP values based on the selected dataset
            if data_type == "Clean":
                shap_values = joblib.load(SHAP_VALUES_FILE_CLEAN)
            else:
                shap_values = joblib.load(SHAP_VALUES_FILE_RAW)

            # Reshape the SHAP values if necessary
            shap_values_reshaped = np.array(shap_values).reshape(100, 5)

            # SHAP Summary Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_reshaped, X_test[:100], feature_names=features, show=False)
            st.pyplot(fig)

            # Button for showing explanation
            if st.button("🔍 Show SHAP Value Explanation"):
                shap_explanation()

        except Exception as e:
            st.error(f"An error occurred while loading SHAP values: {str(e)}")

if __name__ == "__main__":
    main()
