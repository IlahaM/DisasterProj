import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load the data from 2 files and merge the results"""
    
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)  
    df = messages.join(categories, how = 'left', lsuffix = 'id', rsuffix = 'id')
    
    return df


def clean_data(df):
    """clean the data by splitting column into several and dropping unnecessary column"""
    
    categories = df['categories'].str.split(";", expand = True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[: -2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop(columns = 'categories', inplace = True)
    
    df = pd.concat([df, categories], axis = 1, sort = False)
    
    df = df[df.duplicated(subset=None, keep='first') == False]
    
    return df


def save_data(df, database_filename):
    """save the data into a database file"""
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disasters', engine, index=False)  
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
	