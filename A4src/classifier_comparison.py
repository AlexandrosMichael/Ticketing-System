from load_data import split_data_labels
from settings import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' This module is concerned with comparing the performance of different classification algorithms on the same data.
This is a part of the additional requirements implemented and it provides some evidence on which classification 
algorithms perform the best for the task at hand. The methodology was inspired by Jeff Delaney at:
https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn
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


def compare_results(classifiers):
    """
    This method is the method which carries out the training of each of the classifiers to be compared
    :param classifiers: the classifiers to be compared
    :return: None
    """
    split_data_dict = split_train_test('tickets.csv')
    X_train = split_data_dict.get('X_train')
    X_test = split_data_dict.get('X_test')
    y_train = split_data_dict.get('y_train')
    y_test = split_data_dict.get('y_test')

    log_cols = ["Classifier", "Accuracy"]
    log = pd.DataFrame(columns=log_cols)

    for c in classifiers:
        c.fit(X_train, y_train)
        y_val_pred = c.predict(X_test)
        print("Results for classifier of type:")
        print(type(c))
        print(confusion_matrix(y_val_pred, y_test))
        print(classification_report(y_test, y_val_pred, target_names=RESPONSE_TEAMS))
        acc = accuracy_score(y_test, y_val_pred)
        log_entry = pd.DataFrame([[c.__class__.__name__, acc * 100]], columns=log_cols)
        log = log.append(log_entry)
    sns.set_color_codes("muted")
    sns.set_context('paper')
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    plt.xlabel('Accuracy %', fontsize=12)
    plt.title('Classifier Accuracy', fontsize=12)
    plt.show()


def head_to_head():
    """
    Method which creates the classifiers to be compared and adds them to a list
    :return: None
    """
    print("Head-to-head comparision of various classification algorithms\n")
    classifiers = []
    clf = MLPClassifier(solver='lbfgs', momentum=0.5, alpha=0.05, hidden_layer_sizes=(7,), activation='relu',
                        early_stopping=True)
    svc = SVC(gamma='auto', C=0.025)
    knn = KNeighborsClassifier(n_neighbors=4)
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier(n_estimators=10)
    gbc = GradientBoostingClassifier()
    classifiers.append(clf)
    classifiers.append(svc)
    classifiers.append(knn)
    classifiers.append(dtc)
    classifiers.append(rfc)
    classifiers.append(gbc)
    compare_results(classifiers)




