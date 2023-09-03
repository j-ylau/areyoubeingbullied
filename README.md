# Are You Being Bullied? - Cyberbullying Detection System

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Tools Used](#tools-used)
- [Objectives Reached](#objectives-reached)
- [Design Decisions](#design-decisions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About The Project

This project aims to create a robust Cyberbullying Detection System that employs machine learning techniques to analyze and predict potential cyberbullying behavior based on textual content. Built primarily in Python, it showcases the use of NLP libraries and machine learning algorithms to construct a model that can distinguish between various types of cyberbullying such as age-based, gender-based, and ethnicity-based bullying, among others.

---

## Getting Started

### Prerequisites

- Python 3.11
- Pip

### Installation

1. Clone the repository
```sh
git clone https://github.com/your_username_/areyoubeingbullied.git
```
2. Create a virtual environment and activate it
```sh
python -m venv venv
source venv/bin/activate
```
3. Install the required packages
```sh
pip install -r requirements.txt
```

---

## Tools Used

- **Python**: The backbone of the project. Utilized for both data preprocessing and model training.
- **Pandas**: Used for data manipulation and analysis.
- **NLTK**: Natural Language Toolkit used for text preprocessing.
- **Scikit-learn**: For creating and evaluating the machine learning models.
- **Joblib**: For saving and loading the trained model.

---

## Objectives Reached

- Developed a balanced dataset considering various types of cyberbullying.
- Engineered features and optimized the RandomForestClassifier for better prediction accuracy.
- Implemented a reusable pipeline using Scikit-learn's Pipeline and FeatureUnion for seamless model training and evaluation.
- Achieved a high classification rate with strong recall and precision across different categories of cyberbullying.

---

## Design Decisions

- **Text Preprocessing**: Extensive text preprocessing including stemming, stop-word removal, and TF-IDF vectorization to capture the essence of the text.
  
- **Classifier Choice**: RandomForestClassifier was chosen for its high accuracy and ability to handle unbalanced datasets effectively.

- **Data Balancing**: Instead of simple under-sampling or over-sampling, a more advanced balancing technique was used to ensure that all classes have equal representation.

- **Modular Code**: Created separate modules for preprocessing and modeling to ensure that the code is reusable and maintainable.

---

## Contributing

If you would like to contribute, please fork the repository and create a pull request. You can also simply drop an issue for any errors or improvements.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

jlau61@calpoly.edu

---

