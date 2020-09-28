'''
This file contains settings that are being used by the program. A settings files allows us to adapt the program to any
changes such as additional response teams, questions etc. The ALLOWED_ERROR variable refers to the error allowed in the
early prediction feature i.e. when allowed error 0.05 an early prediction will be made when the model is 95% certain
of an answer
'''

RESPONSE_TEAMS = ["Emergencies", "Networking", "Credentials", "Datawarehouse", "Equipment"]

RESPONSE_TEAMS_LABEL = [0, 1, 2, 3, 4]

QUESTIONS = ["Request", "Incident", "WebServices", "Login", "Wireless", "Printing", "IdCards", "Staff", "Students"]

ALLOWED_ERROR = 0.05