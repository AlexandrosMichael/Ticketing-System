from load_data import split_data_labels
from settings import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from joblib import dump
import numpy as np

'''
This module is very similar to the train_agent module but it implements additional requirements such as creating graphs,
carrying out a grid search for hyperparameter tuning, and also training an MLP Regressor to predict the number of days
a ticket will need to be dealt with, based on a different dataset.
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


def param_space():
    """
    This method creates the parameter space of the hyperparameters to be tested during the grid search
    :return: a dictionary containing the different values for each hyperparameter
    """
    parameter_space = {
        # number of neurons as seen in lectures notes 'How many units in a hidden layer'
        'hidden_layer_sizes': [(7,), (9,), (18,), (250,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05, 0.1, 0.5],
        'learning_rate': ['constant', 'adaptive'],
        'momentum': [0.01, 0.1, 0.5, 0.9],
    }
    return parameter_space


def param_space_small():
    """
    Same as param_space method but a smaller parameter space is used in order to make the grid search less time
    consuming. Ideal for testing purposes and in cases where time is limited.
    :return:
    """
    parameter_space = {
        # number of neurons as seen in lectures notes 'How many units in a hidden layer'
        'hidden_layer_sizes': [(7,), (9,), (18,)],
        'activation': ['relu', 'logistic'],
        'solver': ['sgd', 'lbfgs'],
        'alpha': [0.05, 0.1, 0.5],
        'learning_rate': ['constant', 'adaptive'],
        'momentum': [0.1, 0.5, 0.9],
    }
    return parameter_space


def hyper_search(mlp, X_train, y_train):
    """
    Method which carries out the grid search. The parameter space is either given by param_space or the
    param_space_small methods (depending on time availability) and it the grid trained neural networks
    :param mlp:
    :param X_train: training set data
    :param y_train: training set labels
    :return:
    """
    #parameter_space = param_space()
    parameter_space = param_space_small()
    clf = GridSearchCV(mlp, parameter_space, cv=5, scoring='accuracy')
    clf_ret = clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    return clf


def get_best_mlp(best_params):
    """
    Method which takes the hyperparameters that had the most success with the classification task and it creates a new
    MLPClassifier with those
    :param best_params: the parameters that the grid search determined that are the best for the classification task
    at hand
    :return: the MLPClassifier
    """
    alpha = best_params.get('alpha')
    activation = best_params.get('activation')
    hidden_layer_sizes = best_params.get('hidden_layer_sizes')
    learning_rate = best_params.get('learning_rate')
    momentum = best_params.get('momentum')
    solver = best_params.get('solver')
    mlp = MLPClassifier(alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_sizes,
                        learning_rate=learning_rate, momentum=momentum, solver=solver)
    return mlp


def train_network(csv_filename):
    """
    This method gets the data and label training and testing set and then instantiates the neural network which is then
    trained after grid search finds the best hyperparameters for this classification task. Furthermore, the method plots
    a graph of accuracy against the training set size, which is the learning curve of the model and it also prints a
    confusion matrix and a classification report showing the performance of the finished model
    :param csv_filename: the name of the file containing the dataset
    :return: the trained MLPClassifier
    """
    split_data_dict = split_train_test(csv_filename)
    X_train = split_data_dict.get('X_train')
    X_test = split_data_dict.get('X_test')
    y_train = split_data_dict.get('y_train')
    y_test = split_data_dict.get('y_test')
    mlp = MLPClassifier(max_iter=100, early_stopping=True)
    clf_grid = hyper_search(mlp, X_train, y_train)
    clf = get_best_mlp(clf_grid.best_params_)
    test_accuracies = []
    for m in range(1, len(X_train)):
        clf.fit(X_train[:m], y_train[:m])
        y_test_predict = clf.predict(X_test)
        test_accuracies.append(accuracy_score(y_test, y_test_predict)*100)
    fig = plt.figure()
    plt.plot(test_accuracies, "b-", linewidth=3)
    plt.xlabel('Training set size', fontsize=8)
    plt.ylabel('Accuracy%', fontsize=8)
    fig.savefig('training_accuracy.png')
    y_val_pred = clf.predict(X_test)
    print("MLP Training Results")
    print(type(clf))
    print(confusion_matrix(y_val_pred, y_test))
    print(classification_report(y_test, y_val_pred))
    return clf


def train_regression_network(csv_filename):
    """
    This method is similar to the one above, but it trains an MLPRegressor for the prediction of the number of days it
    will take for a ticket to be dealt with. It also plots a learning curve and in addition it also displays the
    mean squared error of the model after training is complete.
    :param csv_filename: the name of the file containing the dataset
    :return: the trained MLPRegressor
    """
    split_data_dict = split_train_test(csv_filename)
    X_train = split_data_dict.get('X_train')
    X_test = split_data_dict.get('X_test')
    y_train = split_data_dict.get('y_train')
    y_test = split_data_dict.get('y_test')
    reg = MLPRegressor(solver='lbfgs', momentum=0.5, alpha=0.05, hidden_layer_sizes=(7,), activation='relu',
                       early_stopping=True)
    train_errors, test_errors = [], []
    for i in range(1, len(X_train)):
        reg.fit(X_train[:i], y_train[:i])
        y_train_predict = reg.predict(X_train[:i])
        y_test_predict = reg.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:i]))
        test_errors.append(mean_squared_error(y_test_predict, y_test))
    fig = plt.figure()
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label="Training set")
    plt.plot(np.sqrt(test_errors), 'b-', linewidth=2, label="Testing set")
    plt.xlabel('Training set size', fontsize=10)
    plt.ylabel('MSE', fontsize=10)
    plt.legend(loc='best')
    fig.savefig('training_mse.png')
    y_val_pred = reg.predict(X_test)
    print("MLP Regression Training Results")
    print(type(reg))
    print("Mean Squared Error: " + str(mean_squared_error(y_test, y_val_pred)))
    print("-------------------------------------------------------------------")
    return reg


def write_clf_advanced():
    """
    Method which calls train_network and train_regression_network which produce a trained MLPClassifier and an
    MLPRegressor respectively and then saves the trained models in a file (adv_agent.joblib and time_agent.joblib)
    :return: None
    """
    clf = train_network('tickets.csv')
    reg = train_regression_network('tickets_time.csv')
    dump(clf, 'adv_agent.joblib')
    dump(reg, 'time_agent.joblib')



