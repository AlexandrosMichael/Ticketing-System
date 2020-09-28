from load_data import split_data_labels
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump
import numpy as np

'''
This module is primarily concerned with training the neural network using the training set, and once it's trained,
it saves the model in a file. This neural network is used by the Basic and Intermediate agents to carry out predictions
on which response team a ticket is to be routed to. 
'''


def split_train_test(csv_filename):
    """
    This method splits the data and the labels into training and testing sets which it then returns
    in the form of a dictionary
    :param csv_filename: the name of the csv file which contains the data
    :return: dictionary containing the training and testing data and label split
    """
    data_dict = split_data_labels(csv_filename)
    data = data_dict.get('data')
    labels = data_dict.get('labels')
    X = np.array(data)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    split_data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return split_data_dict


def train_network(csv_filename):
    """
    This method gets the data and label training and testing set and then instantiates the neural network which is then
    trained. A confusion matrix and a classification report is then printed to show the performance of the trained model
    :param csv_filename: the name of the file containing the dataset
    :return: The trained MLPClassifier
    """
    split_data_dict = split_train_test(csv_filename)
    X_train = split_data_dict.get('X_train')
    X_test = split_data_dict.get('X_test')
    y_train = split_data_dict.get('y_train')
    y_test = split_data_dict.get('y_test')
    clf = MLPClassifier(solver='lbfgs', momentum=0.5, alpha=0.05, hidden_layer_sizes=(7,), activation='relu',
                        early_stopping=True)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_test)
    print("MLP Training Results on Basic Agent")
    print(confusion_matrix(y_val_pred, y_test))
    print(classification_report(y_test, y_val_pred))
    print("-------------------------------------------------------------------")
    return clf


def write_clf():
    """
    Method which calls train_network and then saves the trained model in a file
    :return: None
    """
    clf = train_network('tickets.csv')
    dump(clf, 'basic_agent.joblib')




