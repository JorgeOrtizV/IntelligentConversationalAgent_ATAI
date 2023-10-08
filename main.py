from speakeasypy import Speakeasy, Chatroom
import time
from rdflib import Graph

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
                    print("Chatroom: {}; Message: {} - {}; Time: {}".format(room.room_id, message.ordinal, message.message, self.get_time()))
                    #import pdb; pdb.set_trace()
                    # TODO: Improve this conditional, make it robuster
                    if 'SELECT' in message.message: # Naive way of making sure it is a request
                        self.graph.search(message.message)
                        print("\n============QUERY RESULTS===============\n")
                        for element in self.graph.response:
                            print(element[0][0:])


                    room.post_messages("Received your message: {}".format(message.message))
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
        self.graph.parse(source='data/'+graph_dir, format='turtle')
        self.response = ''

    def search(self, query):
        self.response=self.graph.query(query)




if __name__ == "__main__":
    agent = Agent('char-allegro-wok_bot', 'qdoPCWJbUaH9Vw')
    agent.listen()
