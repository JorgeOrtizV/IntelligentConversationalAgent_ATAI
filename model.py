import spacy
import numpy as np
from rapidfuzz import process,fuzz
import pandas as pd

from data import roles_dict, actions_dict, order_dict, prefix_string, query_list

#LOAD DATA

movie_df = pd.read_pickle("./data/movies.pkl")
genre_df = pd.read_pickle("./data/genres.pkl")
name_df  = pd.read_pickle("./data/names.pkl")

movie_dict = dict(zip(movie_df.movie_name, movie_df.uri))
genre_dict = dict(zip(genre_df.genre,genre_df.uri))
name_dict = dict(zip(name_df.name,name_df.uri))


node_dict_map = {
    "<movie_name>":movie_dict,
    "<role>":roles_dict,
    "<action>":actions_dict,
    "<genre>":genre_dict,
    "<name>":name_dict,
    "<order>":order_dict,
                }



nlp_NER = spacy.load("./models/NER/") #load the best model
nlp_NER.add_pipe("merge_entities")

nlp_textcat = spacy.load("./models/textcat/")



def match_entity(entity,entity_dict):
    
    fuzz_match = process.extract(entity, entity_dict.keys(), scorer=fuzz.WRatio, limit=3)
    print(fuzz_match)
    return entity_dict[fuzz_match[0][0]]

def NER_inference(text):

    doc = nlp_NER(text) # input sample text
    preprocessed = " ".join([t.text if not t.ent_type_ else f"<{t.ent_type_.lower()}>" for t in doc])
    ent_dict = {t.text:f"<{t.ent_type_}>" for t in doc if t.ent_type}
    #print(ent_dict)
    
    return {"text":preprocessed,"entities":ent_dict}

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


def inference(input_chat_text):

    ner_res = NER_inference(input_chat_text)
    label = textcat_inference(ner_res["text"])

    query = query_list[int(label)]

    print(ner_res["text"],label,ner_res["entities"])
    #print(query)

    for entity,ent_type in ner_res["entities"].items():

        node_dict = node_dict_map[ent_type]
        print(ent_type,entity)
        matched_node = match_entity(entity,node_dict)
        #print(matched_node)
        if(ent_type == "<action>" or ent_type == "<role>"):
            ent_type = "<action/role>"
        query = query.replace(ent_type,matched_node)
        
    query = query.replace("<number>","10")
    query = query.replace("<genre>","?genre")

    constructed_query = prefix_string + query

    return constructed_query



