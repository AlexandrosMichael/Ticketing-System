from settings import *
from joblib import load
import numpy as np
from pathlib import Path
import os
from train_agent import write_clf
from train_agent_adv import write_clf_advanced


'''
This module is concerned with the requirements of the Intermediate agent. The user is prompted to answer the required
Yes/No questions building a query. Then, a prediction is made by either the basic or advanced agent (depending on the
agent chosen) for as to which response team the ticket will be routed to. The intermediate agent also has the ability to
make an early prediction so that the user does not have to answer all the questions. Furthermore, if the user is unhappy
with their prediction they can inform the program and the program will retrain the neural network and then is able to
take in more queries.
'''

# flag used to hold whether an early prediction has been made
early_prediction_flag = False


# method which is called when an early prediction has been made. Prompts the user to ask whether the correct prediction
# has been made
def early_prediction(predicted_class):
    """
    Method which is called when an early prediction has been made. Prompts the user to ask whether the correct prediction
    has been made
    :param predicted_class: the response team the ticket was predicted to belong to
    :return: true if the user enters y
    """
    user_in = input("Early prediction: " + str(predicted_class + "? Enter y to route ticket to this response team."))
    if user_in == 'y':
        return True
    else:
        return False


def check_probabilities(probabilities):
    """
    Method which checks if the array containing probabilities for the predicted probabilities for each class contains
    an element for which the agent is certain within a margin specified by the ALLOWED_ERROR setting.
    :param probabilities:
    :return: -1 is such probability does not exist in the array, or its index if it is found
    """
    probabilities = probabilities.flatten()
    index = -1
    for i in range(0, len(probabilities)):
        if abs(probabilities[i] - 1.0) <= ALLOWED_ERROR:
            index = i
            break
    return index


def pad_query(query):
    """
    Method which pads an unfinished query with zeros to allow for a prediction to be made by the ANN
    :param query:
    :return:
    """
    extension = []
    extended_query = query.copy()
    for i in range(0, 9 - len(query)):
        extension.append(0)
    extended_query.extend(extension)
    extended_query = np.array(extended_query)
    extended_query = extended_query.reshape(1, -1)
    return extended_query


def get_probability(clf, query):
    """
    Method which returns the class for which receives an unfinished query as a parameter, pads it with zeros and makes
    an early prediction. If the agent is almost certain, (within a margin specified by the ALLOWED_ERROR setting) of a
    :param clf: the trained MLPClassifier
    :param query: the query for which the classifier will make a prediction
    :return: the index of the predicted class
    """
    extended_query = pad_query(query)
    prediction_probability = clf.predict_proba(extended_query)
    prediction_index = check_probabilities(prediction_probability)
    return prediction_index


def build_query(clf):
    """
    Method which provides the text based interface for the ticket routing process. It builds the query, appending a 1 or 0
    depending on the answers of the user. It also calls the appropriate methods so that an early prediction can be made.
    :param clf: the trained MLPClassifier
    :return: the finished query
    """
    query = []
    for i in range(0, len(QUESTIONS)):
        answer = ""
        while (answer != 'y') and (answer != 'n'):
            answer = input(QUESTIONS[i] + "?")
            if answer == 'y':
                query.append(1)
            elif answer == 'n':
                query.append(0)
            else:
                print("Invalid answer. Please answer with y/n.")
            prediction_index = get_probability(clf, query)
            if prediction_index != -1:
                predicted_class = RESPONSE_TEAMS[prediction_index]
                correct_prediction = early_prediction(predicted_class)
                if correct_prediction:
                    global early_prediction_flag
                    early_prediction_flag = True
                    return pad_query(query)
    early_prediction_flag = False
    return query


def get_prediction(query, clf):
    """
    Returns the name of the response team as predicted by the classifier
    :param query: query for which a prediction is to be made
    :param clf: the classifier which makes the prediction
    :return: the name of the response team predicted
    """
    query = np.array(query)
    query = query.reshape(1, -1)
    prediction = RESPONSE_TEAMS[clf.predict(query)[0]]
    return prediction


def get_time_prediction(query, reg):
    """
    Returns the number of days the MLPRegressor predicted the ticket that will be needed for the ticket
    to be dealt with
    :param query: query for which a prediction is to be made
    :param reg: the regressor which makes the prediction
    :return: the number of days predicted
    """
    query = np.array(query)
    query = query.reshape(1, -1)
    prediction = reg.predict(query)[0]
    return prediction


def get_user_feedback(prediction):
    """
    Method which queries the user whether the right prediction has been made
    :param prediction: the prediction of the MLPClassifier
    :return: true if the user answers that the right prediction has indeed be made
    """
    user_feedback = input("Route the ticket to: " + prediction + "?")
    if user_feedback == 'y':
        return True
    else:
        return False


def prompt_correct_route(query):
    """
    Method which asks the user to which response team they wish for their ticket to be routed to in case they are
    unhappy with the prediction made by the MLPClassifier
    :param query: query for which the prediction was made
    :return: a dictionary containing the query that was made and the response team that the user wishes for the ticket
    to be routed to
    """
    print("Where would you like the ticket to be routed to?")
    for i in range(0, len(RESPONSE_TEAMS)):
        print(str(i+1) + " - " + RESPONSE_TEAMS[i])
    ticket_class = input("Enter 1-5: ")
    new_ticket = {
        'query': query,
        'ticket_class': ticket_class
    }
    return new_ticket


def convert_query(query):
    """
    This method converts the query made by the user from 1s and 0s to Yes and No answers for it to be written in the
    csv file containing the training data
    :param query:
    :return: the converted query
    """
    converted_query = []
    for element in query:
        if element == 1:
            converted_query.append("Yes")
        else:
            converted_query.append("No")
    return converted_query


def list_to_string(query_list):
    """
    Utility method which converts a list to a string so that it will be written to a CSV file
    :param query_list: list containing the answers to the yes/no answers
    :return: the string version of the query
    """
    delimiter = ','
    return delimiter.join(query_list)


def update_csv_file(converted_query, csv_filename):
    """
    Method which writes the new query to the dataset when a user is unhappy with a prediction made by the classifier.
    :param converted_query: The query made by the user
    :param csv_filename: The csv file which is to be updated
    :return: None
    """
    path = Path(os.path.abspath(__file__))
    # get to base directory /P4
    base_dir = path.parent.parent
    list_as_string = list_to_string(converted_query)
    print(list_as_string)
    tickets_path = os.path.join(base_dir, csv_filename)
    with open(tickets_path, 'a', newline='') as f:
        f.write('\n')
        f.write(list_as_string)


def retrain_network(query, advanced):
    """
    Method which retrains the network using the updated dataset
    :param query: query that the user was unhappy for its prediction
    :return: None
    """
    new_ticket = prompt_correct_route(query)
    converted_query = convert_query(new_ticket.get('query'))
    ticket_class = RESPONSE_TEAMS[int(new_ticket.get('ticket_class')) - 1]
    converted_query.append(ticket_class)
    update_csv_file(converted_query, 'tickets.csv')
    print("Retraining ANN")
    if advanced:
        write_clf_advanced()
    else:
        write_clf()


def routing_loop():
    """
    Method which carries out the workflow for building the query, making early predictions and retraining the network.
    :return: None
    """
    clf = load("basic_agent.joblib")
    print("Welcome to the ticket routing agent.")
    loop_string = ""
    while loop_string != "q":
        print("Please enter y/n to answer the following questions.")
        query = build_query(clf)
        prediction = get_prediction(query, clf)
        if not early_prediction_flag:
            user_feedback = get_user_feedback(prediction)
            if user_feedback:
                print("Routing ticket to: " + prediction)
            else:
                retrain_network(query, False)
                clf = load("basic_agent.joblib")
        else:
            print("Routing ticket to: " + prediction)
        loop_string = input("Enter q to quit. Any other key to continue with an other key.")
    print("Thank you for using the ticket routing service. Have a great day!")


def routing_loop_adv():
    """
    Method which carries out the workflow for building the query, making early predictions and retraining the network.
    This method also provides the prediction for the number of days that will be needed for the ticket to be dealt with.
    :return: None
    """
    clf = load("adv_agent.joblib")
    reg = load("time_agent.joblib")
    print("Welcome to the advanced ticket routing agent.")
    loop_string = ""
    while loop_string != "q":
        print("Please enter y/n to answer the following questions.")
        query = build_query(clf)
        prediction = get_prediction(query, clf)
        if not early_prediction_flag:
            user_feedback = get_user_feedback(prediction)
            if user_feedback:
                time_prediction = get_time_prediction(query, reg)
                print("Routing ticket to: " + prediction + ". It will approximately take: " +
                      str(int(round(time_prediction))) + " days")
            else:
                retrain_network(query, True)
                clf = load("adv_agent.joblib")
        else:
            time_prediction = get_time_prediction(query, reg)
            print("Routing ticket to: " + prediction + ". It will approximately take: " +
                  str(int(round(time_prediction))) + " days")
        loop_string = input("Enter q to quit. Any other key to continue with an other key.")
    print("Thank you for using the ticket routing service. Have a great day!")
