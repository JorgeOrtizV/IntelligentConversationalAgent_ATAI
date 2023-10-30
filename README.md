# IntelligentConversationalAgent_ATAI
Final Project of the course Advanced Topics in Artificial Intelligence @ UZH

This project consists on an Intelligent agent capable of answering movie-related questions.

### Usage:

Expected input: Close questions. 
- Who is the director of Star Wars: Episode VI - Return of the Jedi? 
- Who is the screenwriter of The Masked Gang: Cyprus? 
- When was "The Godfather" released? 

Sample output:
- I think it is Richard Marquand. 
- The answer suggested by embeddings: Cengiz Küçükayvaz.
- It was released in 1972. 

### Folder structure:
- main.py : Main script. Launches the bot.
- NER\_training.ipynb : Fine-tunning of SpacyNER 
- model.py : Decision making base on SpacyNER and Spacy Text categ. models.
- data.py : Dictionary of auxiliary values

