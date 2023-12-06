import spacy
import numpy as np
from rapidfuzz import process,fuzz
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import pickle

from data import roles_dict, actions_dict, order_dict, prefix_string, query_list, query_spo, predicates_dict

#Load the data

movie_df = pd.read_pickle("./data/movies.pkl")
genre_df = pd.read_pickle("./data/genres.pkl")
name_df  = pd.read_pickle("./data/names.pkl")

movie_dict = dict(zip(movie_df.movie_name, movie_df.uri))
genre_dict = dict(zip(genre_df.genre,genre_df.uri))
name_dict = dict(zip(name_df.name,name_df.uri))

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings_movies = similarity_model.encode(list(movie_dict.keys()), convert_to_tensor=True)
# embeddings_genre = similarity_model.encode(list(genre_dict.keys()), convert_to_tensor=True)
# embeddings_names = similarity_model.encode(list(name_dict.keys()), convert_to_tensor=True)
# embeddings_actions = similarity_model.encode(list(actions_dict.keys()), convert_to_tensor=True)
# embeddings_roles = similarity_model.encode(list(roles_dict.keys()), convert_to_tensor=True)
# embeddings_order = similarity_model.encode(list(order_dict.keys()), convert_to_tensor=True)
# embeddings_predicates = similarity_model.encode(list(predicates_dict.keys()), convert_to_tensor=True)
with open('data/embeddings/embeddings_movies.pkl', 'rb') as f:
    embeddings_movies = pickle.load(f)
with open('data/embeddings/embeddings_genre.pkl', 'rb') as f:
    embeddings_genre = pickle.load(f)
with open('data/embeddings/embeddings_names.pkl', 'rb') as f:
    embeddings_names = pickle.load(f)
with open('data/embeddings/embeddings_actions.pkl', 'rb') as f:
    embeddings_actions = pickle.load(f)
with open('data/embeddings/embeddings_roles.pkl', 'rb') as f:
    embeddings_roles = pickle.load(f)
with open('data/embeddings/embeddings_order.pkl', 'rb') as f:
    embeddings_order = pickle.load(f)
with open('data/embeddings/embeddings_predicates.pkl', 'rb') as f:
    embeddings_predicates = pickle.load(f)

embeddings_map = {
    "<movie>":embeddings_movies,
    "<role>":embeddings_roles,
    "<action>":embeddings_actions,
    "<genre>":embeddings_genre,
    "<name>":embeddings_names,
    "<order>":embeddings_order,
    "<predicate>": embeddings_predicates
                }

node_dict_map = {
    "<movie>":movie_dict,
    "<role>":roles_dict,
    "<action>":actions_dict,
    "<genre>":genre_dict,
    "<name>":name_dict,
    "<order>":order_dict,
    "<predicate>" : predicates_dict
}


#Load the trained models for NER and text-classification

nlp_NER = spacy.load("./models/NER/v3_165/")
nlp_NER.add_pipe("merge_entities")

nlp_textcat = spacy.load("./models/textcat/")


#Function to match entity to entities in KG using fuzzy string similarity 
def match_entity(entity,entity_dict, embedding_dict):
    
    # result = None
    # substr_result = None
    # try:
    #     for key,val in entity_dict.items():
    #         if(entity.lower()==key.lower() and result==None):
    #             result=val
    #             print(f"exact match: {(key,val)}")
    #         elif(entity.lower() in key.lower() and substr_result==None):
    #             substr_result = val
    #             print(f"substring match: {(key,val)}")
    
    #     if(result!=None):
    #         return result
    #     if(substr_result!=None):
    #         return substr_result
    # except Exception as e:
    #     print("Exception in entity matching module: {}".format(e))
    
    # print("fuzzy match")
            
    
    # fuzz_match = process.extract(entity, entity_dict.keys(), scorer=fuzz.WRatio, limit=3)
    # print(fuzz_match)
    # return entity_dict[fuzz_match[0][0]]
    #import pdb; pdb.set_trace()
    embeddings1 = similarity_model.encode([entity], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embedding_dict)
    values, indices = torch.sort(cosine_scores, dim=-1, descending=True)
    values = values[0][:3]
    indices = indices[0][:3]

    entities = list(entity_dict.keys())

    # Print top 3 matches
    similarity_match=''
    for i, idx in enumerate(indices):
        similarity_match+="{} \t\t Score: {:.4f}, ".format(entities[idx], values[i])
    print("Entity similarity: ", similarity_match)
    return entity_dict[entities[indices[0]]]


#Function to detect entities in the text
def NER_inference(text):

    doc = nlp_NER(text) # input sample text
    preprocessed = " ".join([t.text if not t.ent_type_ else f"<{t.ent_type_.lower()}>" for t in doc])
    ent_dict = {t.text:f"<{t.ent_type_}>" for t in doc if t.ent_type}
    #print(ent_dict)
    
    return {"text":preprocessed,"entities":ent_dict}

#Function to classify text into question type and return the label
def textcat_inference(text):

    doc = nlp_textcat(text)

    max_val = -1000
    max_cat = -1
    #print(type(doc2.cats))

    for k,v in doc.cats.items():
        if(v>max_val):
            max_val = v
            max_cat = k

    return(max_cat)

#Function to return sparQL query from input text
def inference(input_chat_text):
    second_option=None
    ner_res = NER_inference(input_chat_text)
    print(ner_res)
    label = textcat_inference(ner_res["text"])

    print("LABEL: ",label)
    detected_predicates = [k for k, v in ner_res["entities"].items() if v=='<predicate>']
    #import pdb;pdb.set_trace()
    # texcat override based on NER
    if 'recommend' in detected_predicates or 'suggest' in detected_predicates:
        label = "9"
        print('label is now 9 - NER override')
    # TODO: Analyze if it is safe to override label here. I am not sure, so let's stick with text cat as per now.
    elif len(detected_predicates) >= 1:
        for pred in detected_predicates:
            if pred in ['release', 'when', 'date', 'year']:
                if label != "2": # Check that this is for the year questions
                    print("Mismatch between NER (label {}) and textcat (label {}). Analyze this".format(2, label))
                    second_option = 2
                else:
                    print('NER and textcat agreement')
            elif pred in ['genre', 'type', 'category']:
                if label != "3": # Check that this is for the year questions
                    print("Mismatch between NER (label {}) and textcat (label {}). Analyze this".format(3, label))
                    second_option = 3
                else:
                    print('NER and textcat agreement')
            elif pred in ['rated', 'rating', 'review', 'score']:
                if label != "4" and label != "5" and label != "6" and label != "8": # Check that this is for the year questions
                    print("Mismatch between NER (label {}) and textcat (label {}). Analyze this".format(3, label))
                    if "<name>" in ner_res["entities"].values() and "<action>" in ner_res["entities"].values():
                        second_option = 6
                    elif "<genre>" in ner_res["entities"].values():
                        second_option = 5
                    else:
                        second_option = 4
                else:
                    print('NER and textcat agreement')
    # TODO: Naive way to check if recommendation, need to make it as last label
    if(label=="9"):
        print("Recommendation")
        query_type = "REC"
        ent_list = []
        for entity,ent_type in ner_res["entities"].items():
            if(ent_type == "<name>" or ent_type == "<movie>" or ent_type == "<genre>"):
                ent_list.append(entity)
                print(entity)
            else:
                print(ent_type)

        return {"query_type": query_type,"entity_list":ent_list}

    else:
        query_type = "QUE"


        query = query_list[int(label)]
        s,p,o = query_spo[int(label)]

        print(ner_res["text"],label,ner_res["entities"])    
        
        #Replace entities in the sparQL query with the detected entities

        for entity,ent_type in ner_res["entities"].items():

            node_dict = node_dict_map[ent_type]
            embedding_dict = embeddings_map[ent_type]
            print(ent_type,entity)
            matched_node = match_entity(entity,node_dict, embedding_dict)

            if(ent_type == "<action>" or ent_type == "<role>"):
                ent_type = "<action/role>"
            s = s.replace(ent_type,matched_node)
            p = p.replace(ent_type,matched_node)
            o = o.replace(ent_type,matched_node)
            query = query.replace(ent_type,matched_node)

        #Default values if user hasn't provided values 
        query = query.replace("<number>","10")
        query = query.replace("<genre>","?genre")

        constructed_query = prefix_string + query
        if ('<movie>' in constructed_query or '<action/role>' in constructed_query or '<name>' in constructed_query) and second_option!=None:
            query = query_list[second_option]
            s,p,o = query_spo[second_option]

            print(ner_res["text"],label,ner_res["entities"])    
            
            #Replace entities in the sparQL query with the detected entities

            for entity,ent_type in ner_res["entities"].items():

                node_dict = node_dict_map[ent_type]
                embedding_dict = embeddings_map[ent_type]
                print(ent_type,entity)
                matched_node = match_entity(entity,node_dict, embedding_dict)

                if(ent_type == "<action>" or ent_type == "<role>"):
                    ent_type = "<action/role>"
                s = s.replace(ent_type,matched_node)
                p = p.replace(ent_type,matched_node)
                o = o.replace(ent_type,matched_node)
                query = query.replace(ent_type,matched_node)

            #Default values if user hasn't provided values 
            query = query.replace("<number>","10")
            query = query.replace("<genre>","?genre")

            constructed_query = prefix_string + query

        return {"query_type":query_type,"query":constructed_query,"spo":(s,p,o)}



