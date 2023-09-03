from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib  # Importing joblib directly
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

class BullyingClassifier:
    def __init__(self):
        self.model_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(preprocessor=preprocess, max_features=1000)),
                ('count', CountVectorizer(preprocessor=preprocess))
            ])),
            ('classifier', RandomForestClassifier(n_estimators=100))
        ])

    def train(self, X_train, y_train):
        self.model_pipeline.fit(X_train, y_train)

    def evaluate(self, X_val, y_val):
        val_predictions = self.model_pipeline.predict(X_val)
        report = classification_report(y_val, val_predictions)
        print(report)

    def save_model(self, filepath):
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filepath)
        joblib.dump(self.model_pipeline, model_path)

    def load_model(self, filepath):
        model_path = os.path.join('models', filepath)
        self.model_pipeline = joblib.load(model_path)

class AdvancedBullyingClassifier(BullyingClassifier):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100)
