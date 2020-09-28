import sys
from train_agent import write_clf
from train_agent_adv import write_clf_advanced
from route_ticket import routing_loop, routing_loop_adv
from classifier_comparison import head_to_head

'''This is the main module of the program. It takes in a parameter which determines which agent is used
 i.e whether it's the basic agent (Bas), the intermediate agent (Int) or the advanced agent (Avd). Furthermore,
 an additional option is provided which can be chosen to compare different classification algorithms head-to-head
 (Comp)'''

# command line argument
agent = sys.argv[1]

if agent == "Bas":
    write_clf()
elif agent == "Int":
    write_clf()
    routing_loop()
elif agent == "Adv":
    write_clf_advanced()
    routing_loop_adv()
elif agent == "Comp":
    head_to_head()




