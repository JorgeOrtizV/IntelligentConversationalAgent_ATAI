import spacy
import numpy as np
from rapidfuzz import process,fuzz
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import pickle

from data import roles_dict, actions_dict, order_dict, prefix_string, query_list, query_spo, predicates_dict, crowd_dict

#Load the data

movie_df = pd.read_pickle("./data/movies.pkl")
genre_df = pd.read_pickle("./data/genres.pkl")
name_df  = pd.read_pickle("./data/names.pkl")
final_crowd_df = pd.read_pickle("./data/crowd_data_preprocessed.pkl")

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
# embeddings_crowd = similarity_model.encode(list(crowd_dict.keys()), convert_to_tensor=True)
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
with open('data/embeddings/embeddings_crowd.pkl', 'rb') as f:
    embeddings_crowd = pickle.load(f)
# with open('data/embeddings/embeddings_crowd.pkl', 'wb') as f:
#     pickle.dump(embeddings_crowd, f)

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
# NER for predicates
nlp_NER2 = spacy.load("./models/NER2/v2_66/")
nlp_NER2.add_pipe("merge_entities")

nlp_textcat = spacy.load("./models/textcat_v3_13_99/")


#Function to match entity to entities in KG using fuzzy string similarity 
def match_entity(entity,entity_dict, embedding_dict):
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
    return entity_dict[entities[indices[0]]], values[0]


#Function to detect entities in the text
def NER_inference(text):

    doc = nlp_NER(text) # input sample text
    preprocessed = " ".join([t.text if not t.ent_type_ else f"<{t.ent_type_.lower()}>" for t in doc])
    ent_dict = {t.text:f"<{t.ent_type_}>" for t in doc if t.ent_type}
    #print(ent_dict)
    
    return {"text":preprocessed,"entities":ent_dict}

def NER2_inference(text):

    doc = nlp_NER2(text) # input sample text
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
    ner_preds = NER2_inference(input_chat_text)
    print(ner_res)
    print(ner_preds)
    label = textcat_inference(ner_res["text"])

    print("LABEL: ",label)
    # TODO : Second check extra predicates
    detected_predicates = [k for k, v in ner_res["entities"].items() if v=='<predicate>']
    extra_predicates = [k for k, v in ner_preds["entities"].items() if v=='<predicate>']
    crowd_predicates = [k for k, v in ner_preds["entities"].items() if v!='<predicate>']
    crowd_predicates.extend([k for k, v in ner_res["entities"].items() if v=='<role>' or k=='genre'])
    if len(extra_predicates)>0:
        detected_predicates.extend(extra_predicates)
    #import pdb;pdb.set_trace()
    # texcat override based on NER
    if 'recommend' in detected_predicates or 'suggest' in detected_predicates or "Recommend" in detected_predicates:
        label = "10"
        print('label is now 10 - NER override')
    elif 'image' in detected_predicates or 'picture' in detected_predicates or 'photo' in detected_predicates or 'looks like' in detected_predicates or 'look like' in detected_predicates:
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
    if(label=="10"):
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
    elif(label == "9"):
        query_type = "IMG"
        ent_list = []
        for entity,ent_type in ner_res["entities"].items():
            if(ent_type == "<movie>" or ent_type=="<name>"):
                ent_list.append(entity)
                print(entity)
            else:
                print(ent_type)
        return {"query_type": query_type,"entity_list":ent_list, "query":query_list[int(label)]}
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
            matched_node, score = match_entity(entity,node_dict, embedding_dict)

            if(ent_type == "<action>" or ent_type == "<role>" or ent_type == "<predicate>"):
                ent_type = "<action/role>"
            s = s.replace(ent_type,matched_node)
            p = p.replace(ent_type,matched_node)
            o = o.replace(ent_type,matched_node)
            query = query.replace(ent_type,matched_node)

        # Check if the crowd knows something about this.
        crowd = search_crowd(s, p, crowd_predicates)
        if crowd != None:
            return crowd
            

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
                matched_node, score = match_entity(entity,node_dict, embedding_dict)

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

def search_crowd(s, p, crowd_predicates):
    found_s = False
    valid_rows=[]
    idx = 0
    for _,row in final_crowd_df.iterrows():
        if(row["Input1ID"].split(":")[-1] == s.split("/")[-1].replace(">","")):
            found_s = True
            valid_rows.append(idx) 
            if(row["Input2ID"].split(":")[-1] == p.split("/")[-1].replace(">","")):
                print(row["Input1ID"].split(":")[-1],s,row["Input2ID"].split(":")[-1],p)
                
                answer = None
                if(row["Answer_bool"]==False):
                    if(row["Fix_value"]==None):
                        answer = row["Input3ID"]
                    else:
                        answer = row["Fix_value"]
                else:
                    answer = row["Input3ID"]
                
                if answer != None:
                    return{"query_type":"CROWD",
                            "IRA":row["Inter_rater_agreement"],
                            "answer":answer,
                            "no_correct":row["No_correct"],
                            "no_incorrect":row["No_incorrect"]
                            }
        idx+=1
    # Only do this if we are 100% certain that we have s
    if found_s:
        #import pdb;pdb.set_trace()
        for pred in crowd_predicates:
            p, score = match_entity(pred,crowd_dict, embeddings_crowd)
            for _,row in final_crowd_df.iloc[valid_rows].iterrows():
                if(row["Input1ID"].split(":")[-1] == s.split("/")[-1].replace(">","") and
                    row["Input2ID"].split(":")[-1] == p.split("/")[-1].replace(">","")):
                    print(row["Input1ID"].split(":")[-1],s,row["Input2ID"].split(":")[-1],p)
                    
                    answer = None
                    if(row["Answer_bool"]==False):
                        if(row["Fix_value"]==None):
                            answer = row["Input3ID"]
                        else:
                            answer = row["Fix_value"]
                    else:
                        answer = row["Input3ID"]
                    
                    if answer != None:
                        return{"query_type":"CROWD",
                                "IRA":row["Inter_rater_agreement"],
                                "answer":answer,
                                "no_correct":row["No_correct"],
                                "no_incorrect":row["No_incorrect"]
                                }
    return None



