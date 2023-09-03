from flask import Flask, request, jsonify
from preprocessing import preprocess
from model import BullyingClassifier

app = Flask(__name__)
classifier = BullyingClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']
    preprocessed_text = preprocess(input_text)
    prediction = classifier.predict(preprocessed_text)
    return jsonify({'predicted_label': prediction})

if __name__ == '__main__':
    classifier.load_model('../notebooks/model.pkl', '../notebooks/vectorizer.pkl')
    app.run(debug=True)
