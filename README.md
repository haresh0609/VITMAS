# VITMAS
Sure! Here's a sample README file for your Fake News Detection Model project:

---

# Fake News Detection Model

## Project Description
This project aims to develop a Fake News Detection Model using Natural Language Processing (NLP) techniques. The goal is to classify news articles as real or fake based on textual features, helping to identify misleading information and ensure news credibility.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```

3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Dataset
Place your training and testing datasets (`train.csv` and `test.csv`) in the project directory.

## Usage
1. Load and preprocess the data:
   ```python
   import pandas as pd
   import re
   from nltk.corpus import stopwords

   # Load the datasets
   train_df = pd.read_csv('train.csv')
   test_df = pd.read_csv('test.csv')

   # Ensure all entries in the text column are strings and fill NaN values with an empty string
   train_df['text'] = train_df['text'].astype(str).fillna('')
   test_df['text'] = test_df['text'].astype(str).fillna('')

   # Text cleaning function
   def clean_text(text):
       text = re.sub(r'\W', ' ', text)
       text = re.sub(r'\s+', ' ', text)
       text = text.lower()
       text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
       return text

   # Clean the training data
   train_df['cleaned_text'] = train_df['text'].apply(clean_text)

   # Clean the testing data
   test_df['cleaned_text'] = test_df['text'].apply(clean_text)
   ```

2. Feature extraction using TF-IDF Vectorizer:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   tfidf = TfidfVectorizer(max_features=5000)
   X = tfidf.fit_transform(train_df['cleaned_text']).toarray()
   y = train_df['label']

   X_test = tfidf.transform(test_df['cleaned_text']).toarray()
   y_test = test_df['label']
   ```

3. Split the training data into training and validation sets:
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. Hyperparameter tuning using GridSearchCV:
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    param_grid = {'C': [0.1, 1, 10, 100]}
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    # Best parameters
    print(f'Best Parameters: {grid.best_params_}')

    # Train the model with the best parameters
    model = grid.best_estimator_
    ```

5. Evaluate the model on the validation set:
    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Validation Precision: {val_precision}')
    print(f'Validation Recall: {val_recall}')
    print(f'Validation F1 Score: {val_f1}')
    print(classification_report(y_val, y_val_pred))
    ```

6. Evaluate the model on the test set:
    ```python
    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1 Score: {test_f1}')
    print(classification_report(y_test, y_test_pred))
    ```


---
