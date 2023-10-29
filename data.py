roles_dict = {'actor': '<http://www.wikidata.org/prop/direct/P161>',
 'cast': '<http://www.wikidata.org/prop/direct/P161>',
 'cast member': '<http://www.wikidata.org/prop/direct/P161>',
 'director': '<http://www.wikidata.org/prop/direct/P57>',
 'screenwriter': '<http://www.wikidata.org/prop/direct/P58>',
 'producer': '<http://www.wikidata.org/prop/direct/P162>'}

actions_dict = {'acted': '<http://www.wikidata.org/prop/direct/P161>',
 'directed': '<http://www.wikidata.org/prop/direct/P57>',
 'screenwrote': '<http://www.wikidata.org/prop/direct/P58>',
 'wrote': '<http://www.wikidata.org/prop/direct/P58>',
 'written': '<http://www.wikidata.org/prop/direct/P58>',
 'produced': '<http://www.wikidata.org/prop/direct/P162>',
 'featured': '<http://www.wikidata.org/prop/direct/P161>',
 'featuring': '<http://www.wikidata.org/prop/direct/P161>'}

order_dict = {
    "highest":"DESC",
    "highly":"DESC",
    "best":"DESC",
    "top":"DESC",
    "lowest":"ASC",
    "worst":"ASC",
    "bottom":"ASC",
}

prefix_string = "PREFIX ddis: <http://ddis.ch/atai/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX schema: <http://schema.org/> "

query_list = ['SELECT ?y WHERE { <movie_name> <action/role> ?x . ?x rdfs:label ?y}',
 'SELECT ?y WHERE { ?x <action/role> <name> . ?x rdfs:label ?y . ?x wdt:P136 <genre> . }',
 'SELECT ?x WHERE { <movie_name> wdt:P577 ?x}',
 'SELECT ?y WHERE { <movie_name> wdt:P136 ?x . ?x rdfs:label ?y}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}',
 'SELECT ?lbl WHERE { SELECT ?movie ?lbl ?rating WHERE { ?movie wdt:P31 wd:Q11424 . ?movie ddis:rating ?rating . ?movie <action> <name> . ?movie wdt:P136 <genre> . ?movie rdfs:label ?lbl . } ORDER BY <order>(?rating) LIMIT <number>}']

node_dict_map = {
    "<movie_name>":movie_dict,
    "<role>":roles_dict,
    "<action>":actions_dict,
    "<genre>":genre_dict,
    "<name>":name_dict,
    "<order>":order_dict,
                }