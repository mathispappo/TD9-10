import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.probability import FreqDist

# Liste des noms de fichiers
file_names = ['valeursfoncieres/valeursfoncieres-2018.txt', 'valeursfoncieres/valeursfoncieres-2019.txt', 'valeursfoncieres/valeursfoncieres-2020.txt', 'valeursfoncieres/valeursfoncieres-2021.txt', 'valeursfoncieres/valeursfoncieres-2022.txt']

# Création d'une liste vide pour stocker les DataFrames
dataframes = []

# Parcourir chaque fichier et charger les données dans un DataFrame
for file_name in file_names:
    # Charger le fichier en utilisant la fonction read_csv de pandas
    df = pd.read_csv(file_name, delimiter='|', encoding='utf-8', nrows=1000)
    # Ajouter le DataFrame à la liste
    dataframes.append(df)

# Fusionner les DataFrames en un seul DataFrame
merged_df = pd.concat(dataframes)

# Supprimer les colonnes inutiles
columns_to_drop = ['Identifiant de document', 'Reference document', '1 Articles CGI', '2 Articles CGI', '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', 'No disposition', 'No voie', 'B/T/Q', 'Code voie', 'No plan', 'No Volume', '1er lot', 'Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot', '3eme lot', 'Surface Carrez du 3eme lot', '4eme lot', 'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot', 'Nombre de lots', 'Identifiant local']
merged_df.drop(columns=columns_to_drop, inplace=True)

# Supprimer les enregistrements avec des valeurs manquantes
merged_df.dropna(inplace=True)

# Convertir la colonne 'Valeur fonciere' en format numérique
merged_df['Valeur fonciere'] = merged_df['Valeur fonciere'].str.replace(',', '.').astype(float)

# Fractionner les données en sous-ensembles d'apprentissage et de test
# Par exemple, supposons que vous souhaitez utiliser 80% des données pour l'apprentissage et 20% pour le test
train_ratio = 0.8
train_size = int(len(merged_df) * train_ratio)

train_data = merged_df[:train_size]
test_data = merged_df[train_size:]

# Analyse exploratoire des données
# Par exemple, afficher les statistiques descriptives du prix foncier
print(train_data['Valeur fonciere'].describe())

# Afficher les premières lignes du DataFrame fusionné
print(merged_df.head())