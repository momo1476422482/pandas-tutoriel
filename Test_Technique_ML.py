#!/usr/bin/env python
# coding: utf-8

# # Part I : Quel(le) data scientist êtes-vous ?
# ## Contexte de l’analyse
# 
# Elu métier le plus sexy par la Harvard Business Review en octobre 2012, le data scientist représente un profil rare qui exige de nombreuses compétences.
# 
# A partir d'un dataset Aquila, vous réaliserez :
# - un clustering non supervisé afin d'identifier 2 groupes de profils techniques distinctes
# - une prédiction des profils dont le métier n'est pas labellisé
# 
# 
# ## Données
# data.csv contient 6 variables : 
#     - 'Entreprise' correspond à une liste d'entreprises fictive
#     - 'Metier' correspond au métier parmi data scientist, lead data scientist, data engineer et data architecte
#     - 'Technologies' correspond aux compétences maîtrisées par le profil
#     - 'Diplome' correspond à son niveau scolaire (Bac, Master, PhD,...)
#     - 'Experience' correspond au nombre d'années d'expériences
#     - 'Ville' correspond au lieu de travail
#     
# 
# 
# ## Répondez aux questions 
# 
# Bonne chance!

# In[4]:


# Import des libraries classique (numpy, pandas, ...)
import pandas as pd
import numpy as np
import re
import sklearn as sk

import seaborn as sb
from matplotlib import pyplot as plt
plt.style.use('ggplot')


# ### 1) Importer le tableau de données dans un dataframe 

# In[5]:


# Import du dataframe "data.csv"
df = pd.read_csv("data.csv")
print(df.to_string())




# ### 2) Combien y a t-il d'observations dans ce dataset? Y a t-il des valeurs manquantes? 

# In[14]:


print(df.index) 
#Il y a 9582 observations au total dans ce dataset

df.isnull().any()
# Oui il y a des valeurs manquantes


# ### 3) Réaliser l'imputation des valeurs manquantes pour la variable "Experience" avec : 
# - la valeur médiane pour les data scientists
# - la valeur moyenne pour les data engineers

# In[18]:


#Imputation pour Data scientist
df = pd.read_csv("data.csv")
dff=df.loc[df['Metier']=='Data scientist']
dff['Experience']=pd.to_numeric(dff['Experience'], errors='coerce')
scientist_median=dff['Experience'].median()
print(scientist_median)

#Imputation pour Data engineer
dfe=df.loc[df['Metier']=="Data engineer"]
dfe['Experience']=pd.to_numeric(dfe['Experience'], errors='coerce')
engineer_moy=dfe['Experience'].mean()
print(engineer_moy)

'''
#Imputation pour Data architect
dfa=df.loc[df['Metier']=="Data architect"]
dfa['Experience']=pd.to_numeric(dfa['Experience'], errors='coerce')
architect_moy=dfa['Experience'].mean()
print(architect_moy)

#Imputation pour Lead Data scientist
dfl=df.loc[df['Metier']=="Lead data scientist"]
dfl['Experience']=pd.to_numeric(dfl['Experience'], errors='coerce')
lead_moy=dfl['Experience'].mean()
print(lead_moy)'''


dfs=df.query("Metier=='Data scientist'").fillna({"Experience":f"{scientist_median:.1f}"})
dfe=df.query("Metier=='Data engineer'").fillna({"Experience":f"{engineer_moy:.1f}"})
#dfa=df.query("Metier=='Data architect'").fillna({"Experience":f"{architect_moy:.1f}"})
#dfl=df.query("Metier=='Lead data engineer'").fillna({"Experience":f"{lead_moy:.1f}"})
dfr=df.query("Metier!='Data scientist'& Metier!='Data engineer'")
df=pd.concat([dfs,dfe,dfr])
print(df[["Metier","Experience"]].to_string())

print(df.index)


# ### 4) Combien d'années d'expériences ont, en moyenne, chacun des profils : le data scientist, le lead data scientist et le data engineer en moyenne?

# In[39]:


#df = pd.read_csv("data.csv")

dff=df.loc[df['Metier']=='Data scientist']
dff['Experience']=pd.to_numeric(dff['Experience'], errors='coerce')
moy_ds=dff['Experience'].mean()


dff=df.loc[df['Metier']=='Data engineer']
dff['Experience']=pd.to_numeric(dff['Experience'], errors='coerce')
moy_de=dff['Experience'].mean()



dff=df.loc[df['Metier']=='Lead data scientist']
dff['Experience']=pd.to_numeric(dff['Experience'], errors='coerce')
moy_lds=dff['Experience'].mean()


dff=df.loc[df['Metier']=='Data architecte']
dff['Experience']=pd.to_numeric(dff['Experience'], errors='coerce')
moy_da=dff['Experience'].mean()

print(df['Experience'].to_string())
print(f"Data sciensts ont :{moy_ds :.2f} années d'experience en moyenne")
print(f"Data engineers ont :{moy_de :.2f} années d'experience en moyenne")
print(f"Lead Data sciensts ont :{moy_lds :.2f} années d'experience en moyenne")
print(f"Data architectes ont :{moy_da :.2f} années d'experience en moyenne")


# ### 5) Faire la représentation graphique de votre choix afin de comparer le nombre moyen d'années d'expériences pour chaque métier

# In[48]:


df['Experience']=pd.to_numeric(df['Experience'], errors='coerce')

#ax = sb.boxplot(data=df,x="Metier",y="Experience")
bx=sb.distplot(df['Experience'])


# ### 6) Transformer la variable continue 'Experience' en une nouvelle variable catégorielle 'Exp_label' à 4 modalités: débutant, confirmé, avancé et expert
# - Veuillez expliquer votre choix du règle de transformation.  

# In[53]:


import sklearn.cluster

# Convert DataFrame to matrix

drr['Experience']=pd.to_numeric(df['Experience'], errors='coerce')
drr=drr[drr['Experience'].notnull()]
print(drr['Experience'].to_numpy())
# Using sklearn
km = sklearn.cluster.KMeans(n_clusters=4,init='k-means++')
km.fit(drr['Experience'].to_numpy().reshape(-1,1))
# Get cluster assignment labels
labels = km.labels_
print(km.cluster_centers_)

data_km=pd.DataFrame({"labels":labels,"Experience":drr['Experience'].to_numpy().flatten()})


sb.scatterplot(data=data_km,x="labels",y="Experience")


print(labels)
# Format results as a DataFrame
results = pd.DataFrame([dataset.index,labels]).T


# ### 7) Quelles sont les 5 technologies les plus utilisées? Faites un graphique

# In[44]:



print(pd.Series(df['Technologies'].str.split('/').sum()).value_counts())
#print(df['Technologies'].value_counts())

#les 5 technologies les plus utilisées sont : Python, R, SQL,Java et Hadoop


# ### 8) Réaliser une méthode de clustering non supervisée de votre choix pour faire apparaître 2 clusters que vous jugerez pertinents. Donnez les caractéristiques de chacun des clusters.
# -  Justifier la performance de votre algorithme grace à une métrique.
# -  Interpréter votre resultat.  

# In[ ]:





# ### 9) Réaliser la prédiction des métiers manquants dans la base de données par l'algorithme de votre choix
# -  Justifier la performance de votre algorithme grace à une métrique.
# -  Interpréter votre resultat.  

# In[ ]:




