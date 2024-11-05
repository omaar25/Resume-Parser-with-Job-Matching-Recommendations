import joblib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class ModelEvaluation:
    def __init__(self, x_test_file='data/processed/X_test_tfidf.pkl', 
                 y_test_file='data/processed/y_test.pkl'):
        
        self.x_test_file = x_test_file
        self.y_test_file = y_test_file
        self.X_test_tfidf = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_accuracy = 0

    def load_data(self):
        # Load the test data
        with open(self.x_test_file, "rb") as f:
            self.X_test_tfidf = pickle.load(f)
        with open(self.y_test_file, "rb") as f:
            self.y_test = pickle.load(f)
        print("Test data loaded successfully.\n")

    def load_models(self):
        # Load models from the specified paths
        model_paths = {
            'Random Forest': 'model training/rf_classifier.pkl',
            'Logistic Regression': 'model training/logistic_classifier.pkl',
            'XGBoost': 'model training/xgb_classifier.pkl'
        }
        for model_name, model_path in model_paths.items():
            self.models[model_name] = joblib.load(model_path)
            print(f"{model_name} loaded successfully.")

    def evaluate_models(self):
        # Ensure data is loaded before evaluating
        if self.X_test_tfidf is None or self.y_test is None:
            self.load_data()

        best_model_to_save = None
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test_tfidf)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{model_name} Accuracy: {accuracy:.2f}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model_name
                best_model_to_save = model

        joblib.dump(best_model_to_save, f'model evaluation/best_model_{self.best_model}.pkl')
        print(f"Best model: {self.best_model} with accuracy {self.best_accuracy:.2f} \n")


    def display_classification_report(self):
        # Display classification report for the best model
        y_pred = self.models[self.best_model].predict(self.X_test_tfidf)
        report = classification_report(self.y_test, y_pred)
        print(f"Classification report for {self.best_model}:\n{report}")


    def evaluate(self):
        self.load_data()
        self.load_models()
        self.evaluate_models()
        #self.display_classification_report()
