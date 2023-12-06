import time
import re
import rdflib
from rdflib import Graph
import json
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import traceback

from model import inference, match_entity, movie_dict, embeddings_movies, name_dict, genre_dict
from data import human_like_answers,human_like_answers_embeddings, prefix_string

#Declaring some rdflib namespaces
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

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
    
    def entity_similarity(self,entity_list,entity_dict, embedding_dict):
        dist_all = []
        import pdb; pdb.set_trace()
        labels = []
        for entity in entity_list:
            # TODO : Fix this queries
            # TODO : Format response as a list of strings and add them to label.
            # I was planning to obtain a couple of recommendations based on embeddings in case a movie and a person have a same name.
            if entity in name_dict.keys():
                query='SELECT ?y WHERE { ?x <action/role> <name> . ?x rdfs:label ?y . ?x wdt:P136 <genre> . }'
                uri = name_dict[entity]
                query = query.replace('<name>', uri)
                query = query.replace('<action/role>', '<http://www.wikidata.org/prop/direct/P161>')
                query = query.replace("<genre>","?genre")
                response = test_graph.search(prefix_string+query)
            elif entity in genre_dict.keys():
                query='SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY DESC(?rating) LIMIT 10}'
                query = query.replace('<genre>', genre_dict[entity])
                response = test_graph.search(prefix_string+query)

            entity_uri = match_entity(entity,entity_dict, embedding_dict)
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

        labels = []
        count = 0
        for idx in most_likely:
            try:
                labels.append(self.ent2lbl[self.id2ent[idx]])
                count+=1
            except:
                count-=1
            if(count==10):
                break
        return labels
    
def main(message, test_graph):
    try:
        # import pdb; pdb.set_trace()
        inference_res = inference(message)
        query_type= inference_res["query_type"]
        print(query_type)

        # If query is a recommendation query
        if(query_type=="REC"):
            #import pdb; pdb.set_trace()
            #Obtain the movienames from the entities and find 10 simliar movies using entity_similarity()
            entity_names = inference_res["entity_list"]
            print(entity_names)
            recommendations = test_graph.entity_similarity(entity_names,entity_dict=movie_dict, embedding_dict=embeddings_movies)

            # Add to final message response
            final_response = "Here are some recommendations: "
            for i,rec in enumerate(recommendations):
                final_response+=f"\n {rec}"
            
        else:
            #Query is a question

            query = inference_res["query"]
            #Query the Knowledge Graph 
            test_graph.search(query)

            print("\n============ QUERY RESULTS OBTAINED ===============\n")
            response_message = ""
            if(len(test_graph.response)==0):
                #No results obtained from sparql query
                s,p,o = inference_res["spo"]
                if(len(p)==0 or (len(s)==0 and len(o)==0)):
                    final_response = "Unfortunately I could not answer that question. Could you rephrase it? Maybe I can understand it better."
                else:
                    #Try to get approximate answer using embeddings
                    i = np.random.randint(len(human_like_answers_embeddings))             
                    final_response = human_like_answers_embeddings[i]
                    embeddings_res = test_graph.embedding_prediction(head_uri=s,pred_uri=p)
                    final_response = final_response.replace("<answer>",embeddings_res[0][1])
            
            elif(len(test_graph.response)==1):
                response, = test_graph.response
                text = response[0]
                
                response_message = text
                i = np.random.randint(len(human_like_answers))  
                final_response = human_like_answers[i]
            else:       
                response_message+="\n"  
                print(len(test_graph.response))                   
                for index, element in enumerate(list(set(test_graph.response))):
                    text = element[0]
                    response_message += f"{index+1}: {text} \n"

                
                print(response_message)
                i = np.random.randint(len(human_like_answers))  
                final_response = human_like_answers[i]
            final_response = final_response.replace("<answer>",response_message)

        print(final_response)
    except Exception as e:
        print(e)
        print(traceback.format_exc())



if __name__ == "__main__":
    test_graph = KG()
    while(True):
        message = input("Please provide a query: ")
        main(message, test_graph)