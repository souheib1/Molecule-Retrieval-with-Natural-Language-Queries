# Molecule-Retrieval-with-Natural-Language-Queries
ALTeGraD-2023 Data Challenge

Kaggle challenge: https://www.kaggle.com/competitions/altegrad-2023-data-challenge/overview
Team: Baku incorporated

In this challenge, given a textual query and a list of molecules (represented as graphs) without any textual reference information, the aim is to find the molecule corresponding to the query. Hence, we integrated different types of information: the structured knowledge represented by the text and the chemical properties present in the molecular graphs. The pipeline used to handle this task is built by jointly training a text coder and a molecule coder using contrastive learning. Through contrastive learning, the model learns to match similar text-molecule pairs in the learned representation space, while discarding dissimilar pairs.

The performance of your models will be assessed using the label ranking average precision score (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html)

