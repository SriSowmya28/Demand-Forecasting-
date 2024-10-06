import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objs as go

# Load or define the top 10 products DataFrame
# (Replace with actual data source)
top_10_products = pd.DataFrame({'StockCode': ['84077','85123A','85099B','85123A','85123A','85123A','85099B','85099B','85099B','85099B']})

# Placeholder for complete dataset (replace with actual data)
# Ensure that complete_data has columns: ['StockCode', 'InvoiceDate', 'Quantity']
# For now, creating a dummy dataset
np.random.seed(42)
date_rng = pd.date_range(start='2021-01-01', end='2023-12-31', freq='W')
complete_data = pd.DataFrame({
    'InvoiceDate': np.tile(date_rng, 10),
    'StockCode': np.repeat(top_10_products['StockCode'], len(date_rng)),
    'Quantity': np.random.randint(1, 100, size=len(date_rng) * 10)
})

# Function to create lag features for supervised learning
def create_lag_features(data, lag_steps):
    """Create lag features for time series forecasting."""
    df = pd.DataFrame(data)
    for i in range(1, lag_steps + 1):
        df[f'lag_{i}'] = df['Quantity'].shift(i)
    return df.dropna()

# Streamlit app layout
st.title("Demand Forecasting App")
st.write("This app forecasts demand for the top 10 products over the next 15 weeks.")

# Display the top 10 products
st.subheader("Top 10 Products")
st.dataframe(top_10_products)

# User Input for Stock Code
selected_stock_code = st.selectbox("Select a Stock Code to Forecast", top_10_products['StockCode'])

# Proceed if a stock code is selected
if selected_stock_code:
    # Prepare the time series data for the selected product
    product_data = complete_data[complete_data['StockCode'] == selected_stock_code].groupby('InvoiceDate')['Quantity'].sum()
    product_data.index = pd.to_datetime(product_data.index)
    
    # Split the data into training and testing sets
    forecast_steps = 15  # Forecasting horizon (15 weeks)
    train = product_data[:-forecast_steps]  # All data except the last 15 weeks
    test = product_data[-forecast_steps:]  # Last 15 weeks for testing

    # Create supervised learning data with lag features
    lag_steps = 4  # Number of lag weeks to use as features
    train_df = create_lag_features(train, lag_steps)
    test_df = create_lag_features(pd.concat([train[-lag_steps:], test]), lag_steps)

    # Separate features (X) and target (y) for training and testing
    X_train, y_train = train_df.drop(['Quantity'], axis=1), train_df['Quantity']
    X_test, y_test = test_df.drop(['Quantity'], axis=1), test_df['Quantity']

    # Train XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Forecast the next 15 weeks
    forecast = xgb_model.predict(X_test)

    # Plot Historical and Forecast Plot
    forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='W')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Historical Demand'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines+markers', name='Forecasted Demand'))
    fig.update_layout(title=f'Weekly Demand Forecast for {selected_stock_code} (15 Weeks)',
                      xaxis_title='Date', yaxis_title='Quantity')
    st.plotly_chart(fig)

    # Calculate errors
    train_forecast = xgb_model.predict(X_train)
    train_errors = y_train - train_forecast
    test_errors = y_test - forecast

    # Plot Error Histogram for Training Data
    st.subheader("Training Error Distribution")
    fig_train_error = px.histogram(train_errors, nbins=20, title='Training Error Distribution')
    st.plotly_chart(fig_train_error)

    # Plot Error Histogram for Test Data
    st.subheader("Test Error Distribution")
    fig_test_error = px.histogram(test_errors, nbins=20, title='Test Error Distribution')
    st.plotly_chart(fig_test_error)

    # Show evaluation metrics
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    st.write(f"**MAE for Test Data**: {mae:.2f}")
    st.write(f"**RMSE for Test Data**: {rmse:.2f}")
