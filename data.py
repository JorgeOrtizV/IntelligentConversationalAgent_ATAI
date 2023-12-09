roles_dict = {'actor': '<http://www.wikidata.org/prop/direct/P161>',
 'cast': '<http://www.wikidata.org/prop/direct/P161>',
 'cast member': '<http://www.wikidata.org/prop/direct/P161>',
 'director': '<http://www.wikidata.org/prop/direct/P57>',
 'screenwriter': '<http://www.wikidata.org/prop/direct/P58>',
 'producer': '<http://www.wikidata.org/prop/direct/P162>',
 'developer': '<http://www.wikidata.org/prop/direct/P178>', 
 'film editor': '<http://www.wikidata.org/prop/direct/P1040>', 
 'director of photography': '<http://www.wikidata.org/prop/direct/P344>', 
 'film crew member': '<http://www.wikidata.org/prop/direct/P2079>', 
 'choreographer': '<http://www.wikidata.org/prop/direct/P1809>', 
 'art director': '<http://www.wikidata.org/prop/direct/P3174>', 
 'author': '<http://www.wikidata.org/prop/direct/P50>', 
 'presenter': '<http://www.wikidata.org/prop/direct/P371>', 
 'narrator': '<http://www.wikidata.org/prop/direct/P2438>', 
 'animator': '<http://www.wikidata.org/prop/direct/P6942>', 
 'creator': '<http://www.wikidata.org/prop/direct/P170>', 
 'participant': '<http://www.wikidata.org/prop/direct/P710>', 
 'member of the crew of': '<http://www.wikidata.org/prop/direct/P5096>', 
 'voice actor': '<http://www.wikidata.org/prop/direct/P725>', 
 'publisher': '<http://www.wikidata.org/prop/direct/P123>', 
 'musical conductor': '<http://www.wikidata.org/prop/direct/P3300>', 
 'operator': '<http://www.wikidata.org/prop/direct/P137>', 
 'performer': '<http://www.wikidata.org/prop/direct/P175>'
 }

actions_dict = {'acted': '<http://www.wikidata.org/prop/direct/P161>',
 'directed': '<http://www.wikidata.org/prop/direct/P57>',
 'screenwrote': '<http://www.wikidata.org/prop/direct/P58>',
 'wrote': '<http://www.wikidata.org/prop/direct/P58>',
 'written': '<http://www.wikidata.org/prop/direct/P58>',
 'produced': '<http://www.wikidata.org/prop/direct/P162>',
 'featured': '<http://www.wikidata.org/prop/direct/P161>',
 'featuring': '<http://www.wikidata.org/prop/direct/P161>',
 "recorded" : '<http://www.wikidata.org/prop/direct/P57>',
 "appear" : '<http://www.wikidata.org/prop/direct/P161>',
 "direct" : '<http://www.wikidata.org/prop/direct/P57>',
 "produce" : '<http://www.wikidata.org/prop/direct/P178>',
 "filmed" : '<http://www.wikidata.org/prop/direct/P57>',
 "edit" : '<http://www.wikidata.org/prop/direct/P1040>',
 "edited" : '<http://www.wikidata.org/prop/direct/P1040>',
 "shoot" : '<http://www.wikidata.org/prop/direct/P57>',
 #"premiere" : None,
 #"distribute" : None,
 "directing": '<http://www.wikidata.org/prop/direct/P57>',
 "played": '<http://www.wikidata.org/prop/direct/P161>',
 "made": '<http://www.wikidata.org/prop/direct/P57>',
 "authored": '<http://www.wikidata.org/prop/direct/P57>',
 "appears": '<http://www.wikidata.org/prop/direct/P161>'
}

order_dict = {
    "highest":"DESC",
    "highly":"DESC",
    "best":"DESC",
    "top":"DESC",
    "lowest":"ASC",
    "worst":"ASC",
    "bottom":"ASC",
}

predicates_dict = {
    'release': '<http://www.wikidata.org/prop/direct/P577>', 
    'when': '<http://www.wikidata.org/prop/direct/P577>', 
    'date': '<http://www.wikidata.org/prop/direct/P577>', 
    'year': '<http://www.wikidata.org/prop/direct/P577>', 
    'genre': '<http://www.wikidata.org/prop/direct/P136>', 
    'type': '<http://www.wikidata.org/prop/direct/P136>', 
    'category': '<http://www.wikidata.org/prop/direct/P136>', 
    'rated': '<http://www.wikidata.org/prop/direct/P444>', 
    'rating': '<http://www.wikidata.org/prop/direct/P444>', 
    'review': '<http://www.wikidata.org/prop/direct/P444>', 
    'score': '<http://www.wikidata.org/prop/direct/P444>',
    'suggest' : None,
    'recommend' : None
}

prefix_string = "PREFIX ddis: <http://ddis.ch/atai/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX schema: <http://schema.org/> "

query_list = ['SELECT ?y WHERE { <movie> <action/role> ?x . ?x rdfs:label ?y}',
 'SELECT ?y WHERE { ?x <action/role> <name> . ?x rdfs:label ?y . ?x wdt:P136 <genre> . }',
 'SELECT ?x WHERE { <movie> wdt:P577 ?x}',
 'SELECT ?y WHERE { <movie> wdt:P136 ?x . ?x rdfs:label ?y}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie <action> <name> . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 'SELECT ?lbl  WHERE { <movie> wdt:P31 wd:Q11424 . ?movie wdt:P577 ?release_date ?movie rdfs:label ?lbl .FILTER regex(str(?release_date), "<year>") .}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie <action> <name> . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 #'ASK {<name> <action> <movie> . <movie> wdt:P577 ?release_date .<movie> wdt:P136 <genre> FILTER regex(str(?release_date), "<year>") . }'
 'SELECT ?x WHERE {<movie> wdt:P345 ?x .}', # Obtain IMBd ID
 ]

query_spo = [
    ("<movie>","<action/role>",""),
    ("","<action/role>","<name>"),
    ("<movie>","http://www.wikidata.org/prop/direct/P577",""),
    ("<movie>","http://www.wikidata.org/prop/direct/P136",""),
    ("","",""),
    ("","",""),
    ("","",""),
    ("","",""),
    ("","",""),
    ("","",""),
]

human_like_answers = ["Well, the thing you're wondering about is actually <answer>.",
                     "I think the answer is <answer>",
                     "As far as I know, it's <answer>",
                     "The answer to your question is <answer>",
                     "According to my knowledge, it's <answer>",
                     "Here's the answer to your question: <answer>"
                     ]

human_like_answers_embeddings = ["The answer according to my embeddings: <answer>",
                                 "From my embeddings, I think its <answer>",
                                 "According to my calculations, I believe it's <answer>",
                                 "Let me just check my embeddings, yup it's <answer>",                                 
                                 ]

human_like_answers_recommendations = ["Here are some recommendations based on what you asked: "]