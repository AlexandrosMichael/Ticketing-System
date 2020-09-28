import os
import csv
from settings import *
from pathlib import Path

'''
This module is concerned with loading the data from the csv files, and encoding in it in an appropriate way so that it
can be used by the training algorithms. It carries out several procedures such as splitting the data from the labels,
and encoding each training sample so that it is useful to the training algorithms.
'''


def load_tickets(csv_filename):
    """
    This method finds the csv file passed as a parameter and it returns a list of the row data of the csv file
    :param csv_filename: the name of the file whose data is to be loaded
    :return: list containing the row data of the csv file
    """
    path = Path(os.path.abspath(__file__))
    # get to base directory /P4
    base_dir = path.parent.parent
    tickets_path = os.path.join(base_dir, csv_filename)
    data = []
    with open(tickets_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append(row)
            line_count = line_count + 1
    # return a list of the data, each row contains a list of holding the data of each row
    return data


# encodes a single row of the data
def encode_row(row_data, csv_filename):
    """
    This method encodes a single row of the data set. It converts a list containg the answers to the questions to a list
    of 1s and 0s depending on the answers. i.e. for Yes -> 1 and No -> 0
    :param row_data: the data of a single training set row
    :param csv_filename: the name of the file in which the data resides
    :return: the encoded form of the row
    """
    encoded_row = []
    col_count = 0
    for col in row_data:
        # first nine elements are Yes/No answers
        if col_count < 9:
            if col == "Yes":
                encoded_row.append(1)
            else:
                encoded_row.append(0)
        # last element is the response team or time taken
        else:
            if csv_filename == 'tickets_time.csv':
                encoded_row.append(int(col))
            else:
                encoded_row.append(RESPONSE_TEAMS.index(col))
        col_count = col_count + 1
    return encoded_row


# for each row of the raw data, it calls encode_row which encodes a single row of the data
def encode_data(csv_filename):
    """
    This method gets the raw data from the csv file by calling load_tickets, and it then calls encode_row on each row
    to encode it.
    :param csv_filename: the name of the file in which the data resides
    :return: a list containing the encoded form of the data
    """
    encoded_data = []
    raw_data = load_tickets(csv_filename)
    for row in raw_data:
        encoded_data.append(encode_row(row, csv_filename))

    return encoded_data


def split_data_labels(csv_filename):
    """
    Method which splits the training data from the labels i.e. the answers to the questions (X) and the response team
    that the ticket was routed to (y)
    :param csv_filename: the name of the file in which the data resides
    :return: a dictionary containing the data (X) and the labels (y)
    """
    data = []
    labels = []
    encoded_data = encode_data(csv_filename)
    for row in encoded_data:
        data_row = row[:9]
        label_row = row[len(row)-1]
        data.append(data_row)
        labels.append(label_row)
    data_dict = {
        'data': data,
        'labels': labels
    }
    return data_dict


