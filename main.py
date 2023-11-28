from speakeasypy import Speakeasy, Chatroom
import time
import re
import rdflib
from rdflib import Graph
import json
import csv
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from model import inference, match_entity, movie_dict
from data import human_like_answers,human_like_answers_embeddings

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

#Declaring some rdflib namespaces
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

#The agent class
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
            # TODO: ADD TRY BLOCK HERE TO RETRY ON CRASH
            rooms: list[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages("Hi! I'm {}, ask me any question about movies and i'll try my best to answer".format(room.my_alias))
                    room.initiated = True

                for message in room.get_messages(only_partner=True, only_new=True):
                    try:
                        message.message = message.message.replace('\n', ' ')
                        message.message = re.sub(' +', ' ', message.message)
                        print("Chatroom: {}; Message: {} - {}; Time: {}".format(room.room_id, message.ordinal, repr(message.message), self.get_time()))

                        #Perform inference using NER and text classification on the input message to obtain SparQL query
                        inference_res = inference(message.message)
                        query_type= inference_res["query_type"]

                        # If query is a recommendation query
                        if(query_type=="REC"):

                            #Obtain the movienames from the entities and find 10 simliar movies using entity_similarity()
                            entity_names = inference_res["entity_list"]
                            print(entity_names)
                            recommendations = self.graph.entity_similarity(entity_names,entity_dict=movie_dict)

                            # Add to final message response
                            final_response = "Here are some recommendations: "
                            for i,rec in enumerate(recommendations):
                                final_response+=f"\n {rec}"
                            
                        else:
                            #Query is a question

                            query = inference_res["query"]
                            #Query the Knowledge Graph 
                            self.graph.search(query)

                            print("\n============ QUERY RESULTS OBTAINED ===============\n")
                            response_message = ""
                            if(len(self.graph.response)==0):
                                #No results obtained from sparql query
                                s,p,o = inference_res["spo"]
                                if(len(p)==0 or (len(s)==0 and len(o)==0)):
                                    final_response = "Unfortunately I could not answer that question. Could you rephrase it? Maybe I can understand it better."
                                else:
                                    #Try to get approximate answer using embeddings
                                    i = np.random.randint(len(human_like_answers_embeddings))             
                                    final_response = human_like_answers_embeddings[i]
                                    embeddings_res = self.graph.embedding_prediction(head_uri=s,pred_uri=p)
                                    final_response = final_response.replace("<answer>",embeddings_res[0][1])
                            
                            elif(len(self.graph.response)==1):
                                response, = self.graph.response
                                text = response[0]
                                
                                response_message = text
                                i = np.random.randint(len(human_like_answers))  
                                final_response = human_like_answers[i]
                            else:       
                                response_message+="\n"  
                                print(len(self.graph.response))                   
                                for index, element in enumerate(list(set(self.graph.response))):
                                    text = element[0]
                                    response_message += f"{index+1}: {text} \n"

                                
                                print(response_message)
                                i = np.random.randint(len(human_like_answers))  
                                final_response = human_like_answers[i]
                            final_response = final_response.replace("<answer>",response_message)

                        room.post_messages(final_response)                       
                        room.mark_as_processed(message)
                    except Exception as error:
                        print("Problems parsing this message {}".format(message.message))
                        print('Obtained following exception: {}'.format(error))
                        print("Traceback: \n.\n.\n{}".format(traceback.format_exc()))
                        room.post_messages(f"Uh oh, I seem to have encountered an error! Could you please try again or rephrase your question?")
                        room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

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
        
        # load the embeddings
        self.entity_emb = np.load('./data/entity_embeds.npy')
        self.relation_emb = np.load('./data/relation_embeds.npy')
        print("Embeddings loaded!")

        # load the dictionaries
        with open('./data/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open('./data/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        print("Dictionaries loaded!")
        print("Bot active")

    def search(self, query):
        self.response=self.graph.query(query)
    
    def embedding_prediction(self,head_uri,pred_uri,top_n = 1):
        head = self.entity_emb[self.ent2id[rdflib.term.URIRef(head_uri[1:-1])]]
        # relation
        pred = self.relation_emb[self.rel2id[rdflib.term.URIRef(pred_uri[1:-1])]]
        # add vectors according to TransE scoring function.
        lhs = head + pred
        # compute distance to *any* entity
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        # find most plausible entities
        most_likely = dist.argsort()
        # compute ranks of entities
        ranks = dist.argsort().argsort()

        # what would be more plausible occupations?
        res = [(self.id2ent[idx][len(WD):], self.ent2lbl[self.id2ent[idx]], dist[idx], rank+1) for rank, idx in enumerate(most_likely[:10])]

        return res[:top_n]
    
    def entity_similarity(self,entity_list,entity_dict):
        dist_all = []
        for entity in entity_list:
            entity_uri = match_entity(entity,entity_dict)
            ent = self.ent2id[rdflib.term.URIRef(entity_uri[1:-1])]
            # we compare the embedding of the query entity to all other entity embeddings
            dist = pairwise_distances(self.entity_emb[ent].reshape(1, -1), self.entity_emb).reshape(-1)

            same_idx = np.where(dist==0)[0]
            dist[same_idx]=99999
            #print(dist)
            dist_all.append(dist)

        total_dist = np.array([sum(x) for x in zip(*dist_all)])

        # order by plausibility
        most_likely = total_dist.argsort()

        op = pd.DataFrame([
            (
                self.id2ent[idx][len(WD):], # qid
                self.ent2lbl[self.id2ent[idx]],  # label
                total_dist[idx],             # score
                rank+1,                # rank
            )
            for rank, idx in enumerate(most_likely[:10])],
            columns=('Entity', 'Label', 'Score', 'Rank'))
        return op["Label"]
    

            


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
