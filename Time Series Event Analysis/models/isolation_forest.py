import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
class IsolationForestModel:
    def __init__(self, data):
        self.data = data
        self.isolation_forest_model = None
        self.lof_model = None
        self.events_isolation_forest = None
        self.events_lof = None

    def fit_models(self, contamination=0.01):
        # Fit Isolation Forest model
        X = self.data['Close'].values.reshape(-1, 1)
        self.isolation_forest_model = IsolationForest(contamination=contamination, random_state=42)
        self.isolation_forest_model.fit(X)

        # Predict outliers (events) using Isolation Forest
        self.data['Event_IsolationForest'] = self.isolation_forest_model.predict(X)
        self.data['Event_IsolationForest'] = self.data['Event_IsolationForest'].apply(lambda x: 1 if x == -1 else 0)
        self.events_isolation_forest = self.data[self.data['Event_IsolationForest'] == 1]

        # Fit Local Outlier Factor model
        self.lof_model = LocalOutlierFactor(contamination=contamination)
        self.data['Event_LOF'] = self.lof_model.fit_predict(X)
        self.data['Event_LOF'] = self.data['Event_LOF'].apply(lambda x: 1 if x == -1 else 0)
        self.events_lof = self.data[self.data['Event_LOF'] == 1]

    def evaluate_models(self, true_labels):
        # Evaluate Isolation Forest model
        accuracy_isolation_forest = accuracy_score(true_labels, self.data['Event_IsolationForest'])
        classification_report_isolation_forest = classification_report(true_labels, self.data['Event_IsolationForest'])

        # Evaluate Local Outlier Factor model
        accuracy_lof = accuracy_score(true_labels, self.data['Event_LOF'])
        classification_report_lof = classification_report(true_labels, self.data['Event_LOF'])

        return {
            'accuracy_isolation_forest': accuracy_isolation_forest,
            'classification_report_isolation_forest': classification_report_isolation_forest,
            'accuracy_lof': accuracy_lof,
            'classification_report_lof': classification_report_lof
        }

    def get_events(self):
        return self.events_isolation_forest, self.events_lof

    def display_events(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['Close'], label='Signal', color='blue')

        if not self.events_isolation_forest.empty:
            ax.scatter(self.events_isolation_forest.index, self.events_isolation_forest['Close'],
                       color='red', marker='o', label='Isolation Forest Events')

        if not self.events_lof.empty:
            ax.scatter(self.events_lof.index, self.events_lof['Close'],
                       color='green', marker='s', label='Local Outlier Factor Events')

        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.set_title('Event Detection using Isolation Forest and LOF')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    csv_file_path = r"F:\Pycharm Central Zone\Time Series Event Analysis\data_collection\stock_data.csv"
    stock_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

    np.random.seed(42)
    stock_data['True_Event'] = np.random.choice([0, 1], size=len(stock_data))

    isolation_forest_model = IsolationForestModel(stock_data)

    # Assuming you have a ground truth for events (labels) in a separate column named 'True_Event'
    true_labels = stock_data['True_Event']  # Replace with your actual column name

    isolation_forest_model.fit_models(contamination=0.01)

    events_isolation_forest, events_lof = isolation_forest_model.get_events()

    print("Isolation Forest Events:")
    print(events_isolation_forest)

    print("\nLocal Outlier Factor Events:")
    print(events_lof)

    isolation_forest_model.display_events()

    # Evaluate the models
    evaluation_results = isolation_forest_model.evaluate_models(true_labels)
    print("\nEvaluation Results:")
    for key, value in evaluation_results.items():
        print(f"{key}: {value}")
