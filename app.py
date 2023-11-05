from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('bist100_lstm_model.h5')

# Define a function to prepare input data
def prepare_input_data(data, num_data_points=30):
    data = data[['Close']].values
    data = data[-num_data_points:]  # Use the most recent 'num_data_points' data points
    data = data.reshape(1, num_data_points, 1)  # Reshape for model input
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        stock_symbol = request.form['stock_symbol']
        end_date = request.form['end_date']
        start_date = request.form['start_date']

        data = yf.download(stock_symbol, start=start_date, end=end_date)


        # Prepare input data for prediction
        X = prepare_input_data(data)

        # Use the LSTM model to make price predictions for the next 15 days
        predicted_prices = []
        real_prices = data['Close'].tolist()
        current_date = datetime.strptime(end_date, '%Y-%m-%d')

        for _ in range(15):
            # Predict the next day's price
            prediction = model.predict(X)
            predicted_prices.append(prediction[0][0])

            # Update input data for the next prediction
            X = np.roll(X, shift=-1)
            X[0][-1] = prediction[0][0]

            # Update the current date for the next day
            current_date += timedelta(days=1)

        # Calculate the scaling factor
        scaling_factor = real_prices[0] / predicted_prices[0]

        # Scale the predicted prices to match the scale of real prices
        scaled_predicted_prices = [price * scaling_factor for price in predicted_prices]

        # Create a list of dates for the next 15 days
        next_5_days = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=5)

        # Create a Plotly line chart for scaled predicted and real prices
        fig = go.Figure()
        fig = fig.add_scatter(x=next_5_days, y=scaled_predicted_prices, mode='lines', name='Predicted Prices')
        fig.add_scatter(x=data.index, y=real_prices, mode='lines', name='Real Historical Prices')
        fig.update_layout(
            title=f'Stock Price and Predictions for {stock_symbol}',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        chart = fig.to_html(full_html=False)

        return render_template('index.html', chart=chart)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
