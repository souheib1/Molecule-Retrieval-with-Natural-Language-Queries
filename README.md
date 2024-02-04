# Molecule-Retrieval-with-Natural-Language-Queries
ALTeGraD-2023 Data Challenge
 
Kaggle challenge: https://www.kaggle.com/competitions/altegrad-2023-data-challenge/overview <br>
Team: Baku incorporated
    - [Souheib Ben Mabrouk](https://github.com/souheib1)
    - [Omar Bensaid](https://github.com/Hyperyop)
    - [Heni Soula](https://github.com/HeniSoula)


In this challenge, given a textual query and a list of molecules (represented as graphs) without any textual reference information, the aim is to find the molecule corresponding to the query. Hence, we integrated different types of information: the structured knowledge represented by the text and the chemical properties present in the molecular graphs. The pipeline used to handle this task is built by jointly training a text encoder and a molecule encoder using contrastive learning. Through contrastive learning, the model learns to match similar text-molecule pairs in the learned representation space, while discarding dissimilar pairs.

![final_model](https://github.com/souheib1/Molecule-Retrieval-with-Natural-Language-Queries/assets/73786465/6ffe4fea-4c72-4253-bb01-795ebb5e6f35)


The performance of your models will be assessed using the label ranking average precision score (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html)

