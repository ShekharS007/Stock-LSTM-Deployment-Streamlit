import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import altair as alt

# Load the pre-trained model
model = load_model("regressor.keras")

# Function to predict the next day's stock price
def predict_next_day(data, sc):
    """
    Predicts the next day's stock price based on input data.
    :param data: Preprocessed test data (scaled and transformed)
    :param sc: MinMaxScaler used for scaling
    :return: Predicted price for the next day
    """
    last_100_days = data[-100:]  # Take the last 100 days as input
    last_100_days = np.reshape(last_100_days, (1, last_100_days.shape[0], 1))
    predicted_price = model.predict(last_100_days)
    return sc.inverse_transform(predicted_price)[0][0]  # Inverse scale the prediction

# Function to get the next trading day
def get_next_trading_day(current_date):
    """
    Adjusts the given date to find the next trading day.
    Skips weekends and known public holidays (simplified for weekends).
    """
    next_day = current_date + pd.Timedelta(days=1)
    while next_day.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
        next_day += pd.Timedelta(days=1)
    return next_day

# Streamlit app setup
st.title("Stock Market Analysis & Prediction App 📈")

st.write("""
Analyze stock prices interactively using historical data from Yahoo Finance!
For Indian stocks, use the `.NS` suffix for NSE and `.BO` suffix for BSE.
Example: `RELIANCE.NS` or `TCS.BO`
""")

# User inputs
ticker_symbol = st.text_input("Enter the stock ticker symbol (e.g., MSFT, RELIANCE.NS (For Indian Companies))", "AAPL")
starting_date = st.date_input("Enter the starting date", value=pd.to_datetime("2021-01-01"))
ending_date = st.date_input("Enter the ending date", value=pd.to_datetime("today"))

# Determine the currency based on the ticker symbol
if ticker_symbol.upper().endswith((".NS", ".BO")):
    currency_symbol = "₹"  # Indian Rupee
else:
    currency_symbol = "$"  # Default to USD

# Validate date input
if ending_date <= starting_date:
    st.error("Ending date must be later than the starting date.")
else:
    # Fetch stock data
    try:
        with st.spinner("Fetching data..."):
            ticker_data = yf.Ticker(ticker_symbol)
            hist = ticker_data.history(start=starting_date, end=ending_date)

        if hist.empty:
            st.error("No data found for the given ticker symbol and date range.")
        else:
            # Display stock data
            st.subheader(f"Stock Data for {ticker_symbol} from {starting_date} to {ending_date}")
            st.write(hist)

            # Display stock summary
            st.write("### Stock Summary")
            st.write(f"**Highest Closing Price:** {currency_symbol}{hist['Close'].max():,.2f}")
            st.write(f"**Lowest Closing Price:** {currency_symbol}{hist['Close'].min():,.2f}")
            st.write(f"**Average Closing Price:** {currency_symbol}{hist['Close'].mean():,.2f}")

            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Volume")
                st.line_chart(hist['Volume'])
            with col2:
                st.write("### Closing Price")
                st.line_chart(hist['Close'])

            # Today's stock price variation with Altair
            st.write("### Today's Stock Price Change (Minute-by-Minute Trend)")
            try:
                today_data = ticker_data.history(period="1d", interval="1m")  # Minute-wise intraday data
                if not today_data.empty:
                    # Calculate the average price per minute
                    today_data['Average Price'] = (today_data['Open'] + today_data['Close']) / 2
                    today_data.reset_index(inplace=True)  # Reset index for Altair compatibility

                    # Create Altair chart
                    price_chart = alt.Chart(today_data).mark_line().encode(
                        x=alt.X('Datetime:T', title='Time'),
                        y=alt.Y('Average Price:Q', title=f'Price ({currency_symbol})', 
                                scale=alt.Scale(domain=[today_data['Average Price'].min() - 5,
                                                         today_data['Average Price'].max() + 5]))
                    ).properties(
                        width=700,
                        height=400,
                        title="Today's Stock Price Change (Minute-by-Minute Trend)"
                    )

                    # Render chart
                    st.altair_chart(price_chart)
                else:
                    st.warning("Today's stock price data is not available.")
            except Exception as e:
                st.warning(f"Could not retrieve today's data: {e}")

            # Prepare data for prediction
            dataset = hist['Close'].values.reshape(-1, 1)
            sc = MinMaxScaler(feature_range=(0, 1))
            dataset_scaled = sc.fit_transform(dataset)

            if len(dataset_scaled) >= 100:  # Ensure sufficient data for prediction
                # Get today's price
                today_price = dataset[-1][0]

                # Predict for the next trading day
                next_trading_day = get_next_trading_day(pd.to_datetime(ending_date))
                next_day_price = predict_next_day(dataset_scaled, sc)

                # Display today's price and predicted value
                st.write("### Predicted Stock Price")
                prediction_df = pd.DataFrame({
                    "Date": ["Today", next_trading_day.strftime('%Y-%m-%d')],
                    "Price": [f"{currency_symbol}{today_price:,.2f}", f"{currency_symbol}{next_day_price:,.2f}"]
                })
                st.write(prediction_df)

            else:
                st.warning("Not enough data for prediction. Please select a larger date range.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
