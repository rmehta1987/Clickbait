# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 17:42:58 2017

@author: rmeht
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# Function to convert a raw title to a string of words
# The input is a single string and the output is a single string of processed titles
def title_to_words( raw_title ):
    
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_title) 
    
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # 3. Remove stopwords, first convert stopword lists into a set for performance
    stops = set(stopwords.words("english"))                  
    
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    
    # 5. Join the words back into one string separated by space and return the result.
    return( " ".join( meaningful_words ))  


data = []

file = open('C:\\Users\\rmeht\\Dropbox\\Algorithms\\Clickbait\\clickbait17-train-170331\\clickbait17-train-170331\\instances.json')

for line in file:
    data.append(json.loads(line))

pandadata = pd.DataFrame(data)

file.close()

file = open('C:\\Users\\rmeht\\Dropbox\\Algorithms\\Clickbait\\clickbait17-train-170331\\clickbait17-train-170331\\truth.jsonl')

data = []

for line in file:
    data.append(json.loads(line))

pandaLdata = pd.DataFrame(data)

#Title Labels (clickbait/not-clickbait)
tlabel = pandaLdata['truthClass']

#processed titles
ptitles = []

for i in range(0,len(pandadata)):
    ptitles.append(title_to_words(pandadata['targetTitle'][i]))


#Bag of Words tool
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 8000) 

#fit the training data into a bag of words model
train_data_features = vectorizer.fit_transform(ptitles)

#convert model to array
train_data_features = train_data_features.toarray()


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, tlabel )

fimportant = forest.feature_importances_ 
