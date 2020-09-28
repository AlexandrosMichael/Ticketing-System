import csv
import os
import random
from pathlib import Path

'''
This module is used to create the data set which is used to trained the MLP Regressor which is used to predict the 
number of days that a ticket will needs to be dealt with. This is not part of the agent's workflow but it is insightful
to see how the dataset was curated. The dataset tickets_no_y.csv is the same as the tickets.csv file but without the 
last column, the Response team, and the resulting csv file is the tickets_time.csv file which has a new column, named 
time. 
'''


def calculate_time_value(row):
    """
    This method calculates value of the Time column in each training sample. This method assumes that the more aspects
    i.e. Request, Student, Staff, Web Services etc. a query is concerned with, the more time the ticket will need to be
    deal with
    :param row: the training sample containing Yes, No answers.
    :return: the calculated time value for the training sample (row)
    """
    # typical number of days
    time_values = [1, 2, 1, 2, 1, 1, 3, 1, 1]
    total_time_val = 0
    for i, val in enumerate(row):
        if val == 'Yes':
            total_time_val = total_time_val + time_values[i]
    return total_time_val


def random_time_value():
    """
    This method can be used as an alternative way to calculate the value of the Time column of each training sample.
    It returns a random integer between (and including) 1 and 10
    :return: None
    """
    return random.randint(1, 10)


def load_tickets():
    """
    This method opens the csv files and then carries out the necessary processes to create the new dataset that is to
    be used to train the MLP Regressor
    :return: None
    """
    path = Path(os.path.abspath(__file__))
    # get to base directory /P4
    base_dir = path.parent.parent
    tickets_path = os.path.join(base_dir, 'tickets_no_y.csv')
    tickets_out_path = os.path.join(base_dir, 'tickets_time.csv')
    with open(tickets_path, 'r') as csv_in:
        with open(tickets_out_path, 'w') as csv_out:
            writer = csv.writer(csv_out, lineterminator='\n')
            reader = csv.reader(csv_in)
            all_data = []
            row = next(reader)
            row.append('Time')
            all_data.append(row)
            for row in reader:
                time_val = calculate_time_value(row)
                #time_val = random_time_value()
                row.append(time_val)
                all_data.append(row)
            writer.writerows(all_data)


load_tickets()
