import pandas as pd
import re
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json  # Import json module

class DataProcessing:
    def __init__(self, resume_file='data/raw/resume.csv', samples_file='data/augmentation/samples.json', output_dir='data/processed'):
        self.resume_file = resume_file
        self.samples_file = samples_file
        self.output_dir = output_dir
        self.data = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        # Load the main resume data
        self.data = pd.read_csv(self.resume_file)
        self.data = self.data.drop(columns=['ID']).drop_duplicates().dropna()

    def add_samples(self):
        samples = self.load_samples()
        bpo_df = pd.DataFrame({'Category': ['BPO'] * len(samples['BPO']), 'Feature': samples['BPO']})
        automobile_df = pd.DataFrame({'Category': ['AUTOMOBILE'] * len(samples['Automobile']), 'Feature': samples['Automobile']})

        self.data = pd.concat([self.data, bpo_df, automobile_df], ignore_index=True)

    def load_samples(self):
        # Load samples from a specified JSON file
        try:
            with open(self.samples_file, 'r') as f:
                samples = json.load(f)
            return samples
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the sample file '{self.samples_file}' exists.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: {e}. Please ensure the sample file is a valid JSON file.")
            return {}

    def balance_data(self, target_count=100):
        # Balance the dataset
        balanced_data = []
        for category in self.data['Category'].unique():
            category_data = self.data[self.data['Category'] == category]
            if len(category_data) < target_count:
                balanced_category_data = resample(category_data, replace=True, n_samples=target_count, random_state=42)
            else:
                balanced_category_data = category_data
            balanced_data.append(balanced_category_data)

        self.data = pd.concat(balanced_data, ignore_index=True)

    def clean_data(self):
        self.data['Feature'] = self.data['Feature'].apply(self.clean_text)

    def clean_text(self, txt):
        # Clean the input text
        cleanText = re.sub('http\S+\s', ' ', txt)
        cleanText = re.sub('#\S+\s', ' ', cleanText)
        cleanText = re.sub('@\S+', ' ', cleanText)
        cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
        cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
        cleanText = re.sub('\s+', ' ', cleanText)
        return cleanText.strip()

    def vectorize_data(self):
        # Vectorize the data using TF-IDF
        X = self.data['Feature']
        y = self.data['Category']
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        self.save_data(X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer)
        

    def save_data(self, X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer):
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "X_train_tfidf.pkl"), "wb") as f:
            pickle.dump(X_train_tfidf, f)
        with open(os.path.join(self.output_dir, "X_test_tfidf.pkl"), "wb") as f:
            pickle.dump(X_test_tfidf, f)
        with open(os.path.join(self.output_dir, "y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)
        with open(os.path.join(self.output_dir, "y_test.pkl"), "wb") as f:
            pickle.dump(y_test, f)

        with open(os.path.join(self.output_dir, "tfidf_vectorizer_categorization.pkl"), "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(os.path.join(self.output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

    def process(self):
        self.load_data()
        self.add_samples() 
        self.balance_data()
        self.clean_data()
        self.vectorize_data()
        print('data processed \n')




