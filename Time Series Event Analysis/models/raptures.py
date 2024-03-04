from ruptures import Pelt
import pandas as pd
import matplotlib.pyplot as plt

class RupturesModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.change_points=None

    def fit_model(self, pen_value=5):
        X = self.data['Close'].values.reshape(-1, 1)
        self.model = Pelt(model='rbf').fit(X)

        self.change_points = self.model.predict(pen=pen_value)

    def get_change_points(self):
        return self.change_points

    def display_change_points(self):

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(self.data.index, self.data['Close'], label='Signal', color='blue')

        valid_change_points = [idx for idx in self.change_points if idx < len(self.data)]
        ax.scatter(self.data.index[valid_change_points], self.data['Close'].iloc[valid_change_points],
                   color='red', marker='o', label='Change Points')

        for idx in valid_change_points:
            ax.axvline(self.data.index[idx], color='green', linestyle='--', alpha=0.7, label='Event Line')

        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.set_title('Change Point Detection in Stock Data')
        ax.legend()
        plt.show()


if __name__ == "__main__":

    csv_file_path = r"F:\Pycharm Central Zone\Time Series Event Analysis\data_collection\stock_data.csv"
    stock_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

    ruptures_model = RupturesModel(stock_data)

    ruptures_model.fit_model(pen_value=10)

    change_points = ruptures_model.get_change_points()


    print("Change Points:", change_points)

    ruptures_model.display_change_points()