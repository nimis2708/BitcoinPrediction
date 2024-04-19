import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tools.eval_measures import rmse
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.figure import Figure


# Main app
st.title("Bitcoin Price Prediction Tool")

# Add a bold header mentioning the importance of the prediction tool
st.markdown("# **Quarterly Predictions of Bitcoin Prices**")

# Add a summary of the importance of the tool
#st.markdown("## **Importance**")
st.write("""
This prediction tool aims to forecast the future price of Bitcoin(ON QUARTERLY BASIS) based on historical data. Understanding the potential price movements of Bitcoin can be crucial for investors, traders, and enthusiasts in making informed decisions about buying, selling, or holding cryptocurrency assets. By leveraging advanced forecasting models, this tool provides valuable insights into the possible direction of Bitcoin prices, enabling users to better navigate the volatile cryptocurrency market.
""")

# Load data
bitcoin_database_open_date = pd.read_csv("/Users/divyamishra/Downloads/bitcoin_open_price_per_day.csv")

# Create tabs for "Forecast" and "Plot Entire Data"
st.sidebar.title("Select an option")
tabs = st.sidebar.radio("Select a radiobutton", ["Plot Entire Data", "Data Grid", "Forecast"])

# Handle each tab separately

if tabs == "Plot Entire Data":
# Convert date to datetime
    bitcoin_database_open_date['date'] = pd.to_datetime(bitcoin_database_open_date['date'])
    
    # Group data by quarters and calculate the mean for each quarter
    bitcoin_database_open_date['quarter'] = bitcoin_database_open_date['date'].dt.to_period('Q')
    grouped_data = bitcoin_database_open_date.groupby('quarter').mean().reset_index()

    # Convert Periods to strings for plotting
    grouped_data['quarter_str'] = grouped_data['quarter'].astype(str)
    
    # Calculate the split point for train and test data
    split_point = int(len(grouped_data) * 0.8)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the first 80% of the data (training data) in blue
    ax.plot(grouped_data['quarter_str'][:split_point], grouped_data['open'][:split_point], label="Training Data", color='blue', marker='o')
    
    # Plot the remaining 20% of the data (test data) in green
    ax.plot(grouped_data['quarter_str'][split_point:], grouped_data['open'][split_point:], label="Test Data", color='green', marker='o')

    # Annotate each data point with its value
    for i, txt in enumerate(grouped_data['open']):
        ax.annotate(f"{txt:.2f}", (grouped_data['quarter_str'][i], grouped_data['open'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=6)

    
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Open Price")
    ax.set_title("Entire Dataset")
    
    # Add legend entries for training and test data
    ax.legend(loc='lower right' )
    
    # Rotate x-axis labels diagonally
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True)
    
    st.pyplot(fig)
    

    # Plotting the bar graph of open prices of each day
    # Add a date range slider for user selection

    st.write("""The below bargraph/section is for user to analyze the bitcoin prices for various range of dates, by applying the date filters on the left side of panel""")
    
    start_date = st.sidebar.date_input("Start Date", min_value=bitcoin_database_open_date['date'].min(), max_value=bitcoin_database_open_date['date'].max(), value=bitcoin_database_open_date['date'].min() if 'date' in bitcoin_database_open_date.columns else None)
    end_date = st.sidebar.date_input("End Date", min_value=bitcoin_database_open_date['date'].min(), max_value=bitcoin_database_open_date['date'].max(), value=bitcoin_database_open_date['date'].max() if 'date' in bitcoin_database_open_date.columns else None)

    # Convert start_date and end_date to datetime64[ns] dtype
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the dataset based on user-selected date range
    filtered_data = bitcoin_database_open_date[(bitcoin_database_open_date['date'] >= start_date) & (bitcoin_database_open_date['date'] <= end_date)]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(filtered_data['date'], filtered_data['open'], width=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Open Price")
    ax.set_title("Bitcoin Open Price - Daily")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    
elif tabs == "Data Grid":
    # Add a sidebar tab for showing the data grid
    st.subheader("Bitcoin Data Grid")
    st.dataframe(bitcoin_database_open_date)

elif tabs == "Forecast":
    # Sidebar options
    selected_model = st.sidebar.selectbox("# *Select Model*", ["SARIMAX", "ARIMA", "Linear Regression"])
    
    st.subheader("Bitcoin Price Forecasting - Select Model in left panel and click on ""Generate Forecast"" below to see predictions")

    # Forecasting function
    def forecast(model_name, data, forecast_period):
        if model_name == "SARIMAX":
            # Implement SARIMAX model
            model = SARIMAX(data['open'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            results = model.fit()
            forecast_values = results.forecast(steps=forecast_period)
        elif model_name == "ARIMA":
            # Implement ARIMA model
            model = ARIMA(data['open'], order=(5, 1, 0))
            results = model.fit()
            forecast_values = results.forecast(steps=forecast_period)
        elif model_name == "Linear Regression":
            # Implement Linear Regression model
            X = np.arange(len(data)).reshape(-1, 1)
            y = data['open']
            model = LinearRegression()
            model.fit(X, y)
            forecast_values = model.predict(np.arange(len(data), len(data) + forecast_period).reshape(-1, 1))
        return forecast_values

    # Display the forecast plot based on the selected model
    if st.button("Generate Forecast"):
        # Convert date to datetime
        bitcoin_database_open_date['date'] = pd.to_datetime(bitcoin_database_open_date['date'])
        
        # Group data by quarters and calculate the mean for each quarter
        bitcoin_database_open_date['quarter'] = bitcoin_database_open_date['date'].dt.to_period('Q')
        grouped_data = bitcoin_database_open_date.groupby('quarter').mean().reset_index()

        # Forecast period
        forecast_period = 4  # Forecasting for the next 4 quarters

        # Get the last quarter in the grouped data
        last_quarter = grouped_data['quarter'].iloc[-1]

        # Perform forecast for the next 4 quarters
        forecast_values = forecast(selected_model, grouped_data, forecast_period)

        # Convert Periods to strings for plotting
        grouped_data['quarter_str'] = grouped_data['quarter'].astype(str)

        # Plotting

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(grouped_data['quarter_str'], grouped_data['open'], label="Actual")

        # Get the next quarters for forecasting
        next_quarters = [(last_quarter + i).strftime('%Y-%m') for i in range(1, forecast_period + 1)]

        # Plot forecast for the next 4 quarters
        forecast_line = ax.plot(next_quarters, forecast_values, marker='o', markersize=8, label="Forecast")

        # Annotate each forecasted data point
        for quarter, forecast_value in zip(next_quarters, forecast_values):
            ax.annotate(f'{forecast_value:.2f}', (quarter, forecast_value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=6)

        ax.set_xlabel("Quarter")
        ax.set_ylabel("Open Price")
        ax.set_title(f"{selected_model} Forecast")
        ax.legend(loc='lower right')
        ax.grid(True)

        # Rotate x-axis labels diagonally
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig)

# Credits section
st.sidebar.title("Credits: Team")
st.sidebar.write("Nimish Misra(Team Lead)")
st.sidebar.write("Anay Gangal")
st.sidebar.write("Hrushikesh Attarde")

#st.sidebar.write("Anay Gangal")
#st.sidebar.write("Hrushikesh Attarde")


