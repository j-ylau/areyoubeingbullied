import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

def stem_words(tokens):
    return [ps.stem(word) for word in tokens]

def tokenize_text(text):
    return word_tokenize(text)

def preprocess(text):
    cleaned_text = clean_text(text)
    tokenized_text = tokenize_text(cleaned_text)
    tokenized_text = remove_stopwords(tokenized_text)
    tokenized_text = stem_words(tokenized_text)
    return ' '.join(tokenized_text)


def balance_data(df, target_column='cyberbullying_type'):
    # Count occurrences of each label
    print("Labels distribution before balancing: ", df[target_column].value_counts())
    
    # Find the size of the smallest class
    min_class_size = df[target_column].value_counts().min()
    
    # Create an empty DataFrame to hold the balanced data
    balanced_df = pd.DataFrame()
    
    # For each label, sample min_class_size rows
    for label in df[target_column].unique():
        sample_df = df[df[target_column] == label].sample(min_class_size, replace=False)
        balanced_df = pd.concat([balanced_df, sample_df])
    
    print("Labels distribution after balancing: ", balanced_df[target_column].value_counts())
    
    return balanced_df



