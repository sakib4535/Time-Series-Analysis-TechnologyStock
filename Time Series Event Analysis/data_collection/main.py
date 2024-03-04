import yfinance as yf
import pandas as pd

def fetch_and_save_data(ticker_symbol, start_date, end_date, output_file):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if 'Date' in stock_data.index.names:
        print("THe Date Column is an index")
    else:
        print("The Date column is not an index")

    stock_data.to_csv(output_file)

if __name__ == "__main__":
    # Set the parameters
    ticker_symbol = "AAPL"
    start_date = "2021-01-01"
    end_date = "2023-01-01"
    output_file = "stock_data.csv"

    # Fetch and save stock data
    fetch_and_save_data(ticker_symbol, start_date, end_date, output_file)