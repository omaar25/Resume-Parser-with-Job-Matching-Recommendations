import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings


class ModelTraining:
    def __init__(self):
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.y_train = None
        self.y_test = None
        self.rf_classifier = None
        self.logistic_classifier = None
        self.xgb_classifier = None

    def load_data(self, base_dir='data/processed'):
        # Define the file paths based on the base directory
        x_train_file = f'{base_dir}/X_train_tfidf.pkl'
        x_test_file = f'{base_dir}/X_test_tfidf.pkl'
        y_train_file = f'{base_dir}/y_train.pkl'
        y_test_file = f'{base_dir}/y_test.pkl'
        
        # Load the data from pickle files
        with open(x_train_file, "rb") as f:
            self.X_train_tfidf = pickle.load(f)

        with open(x_test_file, "rb") as f:
            self.X_test_tfidf = pickle.load(f)

        with open(y_train_file, "rb") as f:
            self.y_train = pickle.load(f)

        with open(y_test_file, "rb") as f:
            self.y_test = pickle.load(f)
        
    def train_models(self):
        warnings.filterwarnings("ignore")

        # Train Random Forest Classifier
        self.rf_classifier = RandomForestClassifier(random_state=42)
        self.rf_classifier.fit(self.X_train_tfidf, self.y_train)
        print('Random Forest Classifier Trained \n')

        # Train Logistic Regression Classifier
        self.logistic_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.logistic_classifier.fit(self.X_train_tfidf, self.y_train)
        print('Logistic Regression Classifier Trained \n')

        # Train XGBoost Classifier
        self.xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.xgb_classifier.fit(self.X_train_tfidf, self.y_train)
        print('XGBoost Classifier Trained')

        print("Models trained successfully!\n")

    def save_models(self, output_dir='model training'):
        # Save the trained models
        joblib.dump(self.rf_classifier, f'{output_dir}/rf_classifier.pkl')
        joblib.dump(self.logistic_classifier, f'{output_dir}/logistic_classifier.pkl')
        joblib.dump(self.xgb_classifier, f'{output_dir}/xgb_classifier.pkl')
        print("Models saved successfully!\n")

    def train(self, base_dir='data/processed', output_dir='model training'):
        self.load_data(base_dir)
        self.train_models()
        self.save_models(output_dir)
