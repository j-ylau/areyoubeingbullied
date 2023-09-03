import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess, balance_data
from model import BullyingClassifier, AdvancedBullyingClassifier

# Load the dataset
dataset = pd.read_csv("data/cyberbullying_dataset.csv")
print(f"Dataset shape after loading: {dataset.shape}")

# Balance the dataset
dataset = balance_data(dataset)
print(f"Dataset shape after balancing: {dataset.shape}")

# Preprocess and split
X = dataset['tweet_text'].apply(preprocess)
y = dataset['cyberbullying_type']
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = BullyingClassifier()
classifier.train(X_train, y_train)

# Or if you want to use the advanced classifier
# classifier = AdvancedBullyingClassifier()
# classifier.train(X_train, y_train)

# Save the model and vectorizer
classifier.save_model('./notebooks/model.pkl', './notebooks/vectorizer.pkl')
