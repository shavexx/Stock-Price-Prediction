# Stock Price Prediction Web Application
# Built with Streamlit, Pandas, Scikit-learn, and Plotly

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Application Header
st.title("📈 Stock Price Prediction")
st.caption("Predict stock closing prices using historical market data and Machine Learning")

# ---------------------------------------------------------
# Sidebar - Data Input & Settings
# ---------------------------------------------------------
st.sidebar.header("📁 Data Source & Options")

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# Stock / Company Name input
company_name = st.sidebar.text_input("Company / Stock Name", value="Sample Corp")

# ---------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------
df = None

# Case 1: User uploaded a file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded CSV loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Case 2: Load default sample dataset if available
else:
    default_data_path = os.path.join("data", "stock_data.csv")
    if os.path.exists(default_data_path):
        df = pd.read_csv(default_data_path)
        st.sidebar.info("Using default dataset from 'data/stock_data.csv'")
    else:
        st.error("No dataset found. Please upload a CSV file or add 'stock_data.csv' to the 'data/' folder.")
        st.stop()

# ---------------------------------------------------------
# Step 2: Data Cleaning & Preprocessing
# ---------------------------------------------------------
# Clean missing values
df = df.dropna()

# Standardize column names (strip whitespace)
df.columns = df.columns.str.strip()

# Check for required columns
required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns in dataset: {', '.join(missing_cols)}")
    st.write("Required columns format: Date, Open, High, Low, Close, Volume")
    st.stop()

# Convert Date column to proper datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort data chronologically by Date
df = df.sort_values(by='Date').reset_index(drop=True)

# ---------------------------------------------------------
# Step 3: Summary Cards (Key Metrics)
# ---------------------------------------------------------
st.subheader(f"📊 {company_name} - Market Summary")

# Calculate summary values
latest_close = df['Close'].iloc[-1]
highest_price = df['High'].max()
lowest_price = df['Low'].min()
avg_close = df['Close'].mean()

# Display metrics in 4 columns
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Closing Price", f"₹ {latest_close:.2f}")
col2.metric("Highest Price", f"₹ {highest_price:.2f}")
col3.metric("Lowest Price", f"₹ {lowest_price:.2f}")
col4.metric("Average Closing Price", f"₹ {avg_close:.2f}")

st.markdown("---")

# ---------------------------------------------------------
# Step 4: Historical Price Chart
# ---------------------------------------------------------
st.subheader("📈 Historical Price Trend")

# Create interactive line chart using Plotly
fig_history = px.line(
    df, 
    x='Date', 
    y='Close', 
    title=f"{company_name} Closing Price Over Time",
    labels={'Close': 'Closing Price (₹)', 'Date': 'Date'}
)
fig_history.update_traces(line_color="#1f77b4", line_width=2)
st.plotly_chart(fig_history, use_container_width=True)

# ---------------------------------------------------------
# Step 5: Dataset Preview
# ---------------------------------------------------------
with st.expander("📋 View Raw Dataset Preview"):
    st.write(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Step 6: Train Machine Learning Model (Linear Regression)
# ---------------------------------------------------------
st.subheader("🤖 Machine Learning Model Performance")

# Features (Input columns) and Target (Output column)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Mean Absolute Error (MAE)", f"₹ {mae:.2f}", help="Average error in price prediction")
metric_col2.metric("R² Score (Accuracy)", f"{r2 * 100:.2f}%", help="Proportion of variance explained by model")

# ---------------------------------------------------------
# Step 7: Actual vs Predicted Price Chart
# ---------------------------------------------------------
st.subheader("🎯 Actual vs Predicted Prices")

# Create comparison DataFrame for test set
comparison_df = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
}).reset_index(drop=True)

# Plot actual vs predicted chart
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(y=comparison_df['Actual Price'], mode='lines+markers', name='Actual Price'))
fig_compare.add_trace(go.Scatter(y=comparison_df['Predicted Price'], mode='lines+markers', name='Predicted Price', line=dict(dash='dash')))
fig_compare.update_layout(
    title="Comparison of Actual and Predicted Closing Prices (Test Data)",
    xaxis_title="Test Sample Index",
    yaxis_title="Closing Price (₹)"
)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# Step 8: Interactive Future Price Prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Future Closing Price")
st.write("Enter market parameter values below to estimate the stock closing price:")

# Pre-populate form with latest known market values
latest_open = float(df['Open'].iloc[-1])
latest_high = float(df['High'].iloc[-1])
latest_low = float(df['Low'].iloc[-1])
latest_vol = float(df['Volume'].iloc[-1])

# Input fields using columns
input_col1, input_col2, input_col3, input_col4 = st.columns(4)

with input_col1:
    input_open = st.number_input("Open Price (₹)", value=latest_open, step=1.0)

with input_col2:
    input_high = st.number_input("High Price (₹)", value=latest_high, step=1.0)

with input_col3:
    input_low = st.number_input("Low Price (₹)", value=latest_low, step=1.0)

with input_col4:
    input_volume = st.number_input("Volume", value=latest_vol, step=1000.0)

# Prediction button
if st.button("🚀 Predict Closing Price", type="primary"):
    # Format input DataFrame for model with matching feature names
    user_data = pd.DataFrame(
        [[input_open, input_high, input_low, input_volume]], 
        columns=['Open', 'High', 'Low', 'Volume']
    )
    
    # Generate prediction using trained model
    predicted_close = model.predict(user_data)[0]
    
    # Display predicted price result
    st.success(f"### Predicted Closing Price: ₹ {predicted_close:.2f}")

st.markdown("---")

# ---------------------------------------------------------
# Educational Disclaimer
# ---------------------------------------------------------
st.warning("⚠️ Note: This prediction is for educational purposes only and should not be considered financial advice.")
