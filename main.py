from speakeasypy import Speakeasy, Chatroom
import time
import re
from rdflib import Graph
from html import escape, unescape
import json

from model import inference

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()
        # Load KG
        self.graph = KG()
    
    def listen(self):
        while True:
            rooms: list[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages("Welcome to {}".format(room.my_alias))
                    room.initiated = True

                for message in room.get_messages(only_partner=True, only_new=True):
                    try:
                        message.message = message.message.replace('\n', ' ')
                        message.message = re.sub(' +', ' ', message.message)
                        print("Chatroom: {}; Message: {} - {}; Time: {}".format(room.room_id, message.ordinal, repr(message.message), self.get_time()))
                        #room.post_messages("Received your message: {}".format(message.message))
                        #import pdb; pdb.set_trace()
                        # TODO: Improve this conditional, make it robuster
                        query = inference(message.message)
                        self.graph.search(query)

                        print("\n============ QUERY RESULTS OBTAINED ===============\n")
                        room.post_messages("Obtained {} results for you query:".format(len(self.graph.response)))
                        #print(self.graph.response)
                        response_message = ""
                        for index, element in enumerate(self.graph.response):
                            text = element[0]
                            response_message += f"{index+1}: {text} \n"
                        #print(response_message)
                        room.post_messages(f"The answer to your question is :\n {response_message}")                       
                        room.mark_as_processed(message)
                    except Exception as error:
                        print("Problems parsing this message {}".format(message.message))
                        print('Obtained following exception: {}'.format(error))
                        room.post_messages(f"Failed to execute SparQL query, obtained the following exception: {error}")
                        room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)


    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())
    

class KG:
    def __init__(self, graph_dir='14_graph.nt'):
        self.graph = Graph()
        # load a graph
        print("Loading graph ...")
        self.graph.parse(source='data/'+graph_dir, format='turtle')
        print("Graph loaded!")
        self.response = ''

    def search(self, query):
        self.response=self.graph.query(query)




if __name__ == "__main__":
    try:
        with open("./credentials.json") as f:
            data = json.load(f)
            username = data["username"]
            password = data["password"]
    except Exception as e:
        print(f"Encountered exception while reading credentials: {e}")

    while(True):
        try:
            agent = Agent(username, password)
            agent.listen()
        except Exception as e:
            print(f"Encountered exception while creating agent: {e}")
            #Maybe add timeout of 5 seconds
