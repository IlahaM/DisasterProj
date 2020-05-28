import sys
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    """load the data from a db file"""
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name = 'Disasters', con = engine)
    X = df['message']
    Y = df.drop(columns = ['idid', 'message', 'original', 'genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    """tokenize text using lemmatizer after normalization and removing stop words"""
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """build model using classification pipeline"""
   
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    
    parameters = {
        #'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_neighbors': [3, 5]
    }
    

    cv = GridSearchCV(pipeline, parameters, cv = 3, n_jobs = -1)

    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """given the category evaluate the model printing report about classification metrics"""
    
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(data = y_pred, columns = Y_test.columns)
    
    for column in category_names:
        print("column name: " + column)
        print(classification_report(Y_test[column], y_pred_df[column]))
    


def save_model(model, model_filepath):
    """save the trained model into a pickle file"""
    
    filename = model_filepath

    with open(filename, 'wb') as file:                  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
	