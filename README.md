# DisasterProject


This project is about classifying the category of the message given about disasters. There are 2 csv data files used, one containing the messages and the other categories.
Data has been cleaned and processed, and classification has been applied to the processed data.

disaster_messages.csv and disaster_categories.csv – data files

process_data.py --- python code to load the data from the csv files and prepare data for further classification. As a result of running this code, processed data will be saved to a database file.

train_classifier.py --- python code to build a classification model, train and test it on the data that has been put into a db file using process_data.py. At the end, it saves the trained model into a pickle file.

run.py --- code to run the web app to use the trained model.

Usage:
1.	Run process_data.py using “python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db” on command line in the folder containing the files.

2.	Then, run train_classifier.py using “python train_classifier.py DisasterResponse.db classifier.pkl” (this will take a while to train the model).

3.	Then, run run.py file using “python run.py”, and while it is running, open a new terminal window and type “env|grep WORK”, and finally, insert spaceid and spacedomain into the places in “https://SPACEID-3001.SPACEDOMAIN”.

