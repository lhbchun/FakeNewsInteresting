# imports

# General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import time
import calendar
import re
import os
import sys
import json

import math
from statistics import variance, median, StatisticsError
from scipy.stats import f as F


# sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# NLP
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ps = PorterStemmer()
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Neural Network
import tensorflow as tf
from tensorflow import keras
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Others
import snscrape.modules.twitter as sntwitter
import snscrape.base

sys.path.insert(0, "../gispy")


def appendSentimentScore(data, targetCol, fileName):

    sentiModel = BertForSequenceClassification.from_pretrained("bert-senti")
    sentiTokenizer = AutoTokenizer.from_pretrained("bert-senti")

    sentiList = []
    targetPrefix = "senti"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"

    if os.path.isfile(targetFile):
        sentiList = pd.read_csv(targetFile)
        return pd.concat([data, sentiList], axis=1)

    for i in data[targetCol]:
        inputs = sentiTokenizer(i, return_tensors="pt", truncation=True)
        result = sentiModel(**inputs).logits.tolist()[0]
        sentiList.append(
            [math.exp(i) / sum([math.exp(i) for i in result]) for i in result]
        )

    sentiList = pd.DataFrame(
        sentiList, columns=["sentiment_" + str(i + 1) for i in range(5)]
    )
    sentiList.to_csv(targetFile, index=False)

    return pd.concat([data, sentiList], axis=1)


def appendArousalScore(data, targetCol, fileName):

    arouModel = RobertaForSequenceClassification.from_pretrained(
        "arousal-english-distilbroberta-base"
    )
    arouTokenizer = RobertaTokenizer.from_pretrained(
        "arousal-english-distilbroberta-base"
    )

    arouList = []
    targetPrefix = "roberta_arousal"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"

    if os.path.isfile(targetFile):
        arouList = pd.read_csv(targetFile)
        return pd.concat([data, arouList], axis=1)

    for i in data[targetCol]:
        inputs = arouTokenizer(i, return_tensors="pt", truncation=True)
        result = arouModel(**inputs).logits.tolist()[0][0]
        arouList.append(result)

    arouList = pd.DataFrame(arouList, columns=["roberta_arousal"])
    arouList.to_csv(targetFile, index=False)

    return pd.concat([data, arouList], axis=1)


def appendDominanceScore(data, targetCol, fileName):

    dominModel = RobertaForSequenceClassification.from_pretrained(
        "dominance-english-distilroberta-base"
    )
    dominTokenizer = RobertaTokenizer.from_pretrained(
        "dominance-english-distilroberta-base"
    )

    dominList = []
    targetPrefix = "roberta_dominance"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"

    if os.path.isfile(targetFile):
        dominList = pd.read_csv(targetFile)
        return pd.concat([data, dominList], axis=1)

    for i in data[targetCol]:
        inputs = dominTokenizer(i, return_tensors="pt", truncation=True)
        result = dominModel(**inputs).logits.tolist()[0][0]
        dominList.append(result)

    dominList = pd.DataFrame(dominList, columns=["roberta_dominance"])
    dominList.to_csv(targetFile, index=False)

    return pd.concat([data, dominList], axis=1)


def appendConcretenessScore(data, targetCol, fileName):

    concrModel = RobertaForSequenceClassification.from_pretrained(
        "concreteness-english-distilroberta-base"
    )
    concrTokenizer = RobertaTokenizer.from_pretrained(
        "concreteness-english-distilroberta-base"
    )

    concrList = []
    targetPrefix = "roberta_concreteness"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"

    if os.path.isfile(targetFile):
        concrList = pd.read_csv(targetFile)
        return pd.concat([data, concrList], axis=1)

    for i in data[targetCol]:
        inputs = concrTokenizer(i, return_tensors="pt", truncation=True)
        result = concrModel(**inputs).logits.tolist()[0]
        concrList.append(np.exp(result[1]) / (np.sum(np.exp(result))))

    concrList = pd.DataFrame(concrList, columns=["roberta_concreteness"])
    concrList.to_csv(targetFile, index=False)

    return pd.concat([data, concrList], axis=1)


def tk(text):
    return word_tokenize(re.sub(r'[^\w\s]', '', str(text).lower())) 

def stopword(tokens):
    result = []
    for w in tokens:
        if w not in stop_words:
            result.append(w)
    return result

def lemma(tokens):
    result = []

    for token in tokens:
        stemmed_word = wnl.lemmatize(token)
        result.append(stemmed_word)
    return result


def valence(tokens):
    
    count = 0
    result = 0
    
    for token in tokens:
        try:
            valence = vad["V.Mean.Sum"][token]
            result += valence
            count += 1
        except KeyError:
            continue
            
    if count > 0:
        return result/count
    return 5

def arousal(tokens):
    
    count = 0
    result = 0
    
    for token in tokens:
        try:
            arousal = vad["A.Mean.Sum"][token]
            result += arousal
            count += 1
        except KeyError:
            continue
            
    if count > 0:
        return result/count
    return 5

def dominance(tokens):
    
    count = 0
    result = 0
    
    for token in tokens:
        try:
            dominance = vad["D.Mean.Sum"][token]
            result += dominance
            count += 1
        except KeyError:
            continue
            
    if count > 0:
        return result/count
    return 5


def appendVAD(data,targetCol, fileName):
    
    targetPrefix = "VADScore_"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"
    
    if os.path.isfile(targetFile):
        VADList = pd.read_csv(targetFile)
        return pd.concat([data, VADList], axis = 1)
    
    Valence = []
    for i in range(len(data)):
        Valence.append(valence(lemma(stopword(tk(data[targetCol][i]))))) 
    Valence = pd.DataFrame(Valence, columns = ["valence"])

    Arousal = []
    for i in range(len(data)):
        Arousal.append(arousal(lemma(stopword(tk(data[targetCol][i]))))) 
    Arousal = pd.DataFrame(Arousal, columns = ["arousal"])
    
    Dominance = []
    for i in range(len(data)):
        Dominance.append(dominance(lemma(stopword(tk(data[targetCol][i]))))) 
    Dominance = pd.DataFrame(Dominance, columns = ["dominance"])
    
    
    VADList = pd.concat([Valence, Arousal, Dominance], axis = 1)
    VADList.to_csv(targetFile, index=False)
    
    
    
    return pd.concat([data, VADList], axis = 1)
    

def appendFrequency(data,targetCol, fileName):
    #Max, Median, Median (Content Word), Min (Content Word)

    freq = pd.read_csv("data/misc/unigram_freq.csv", index_col = 0).reset_index().reset_index()
    freq["index"] = freq["index"] + 1
    rankDict = freq.set_index("word", drop=True).to_dict()["index"]
    freqDict = pd.read_csv("data/misc/unigram_freq.csv", index_col = 0).to_dict()["count"]
    
    targetPrefix = "wordFreq_"
    targetFile = "data/dynamic/" + targetPrefix + "_" + fileName + ".csv"
    
    if os.path.isfile(targetFile):
        df = pd.read_csv(targetFile)
        return pd.concat([data, df], axis = 1)
    
    df = pd.DataFrame()
    
    
    #Max
    x = []
    for i in data[targetCol]:
        tokens = tk(i)
        v = -1
        for j in tokens:
            try:
                if rankDict[j] > v:
                    v = rankDict[j] 
            except KeyError:
                continue
        x.append(v)
    x = pd.DataFrame(x, columns = ["MaxRank"])
        
    df = pd.concat([df, x], axis = 1)
    
    #Median
    x = []
    for i in data[targetCol]:
        tokens = tk(i)
        v = []
        for j in tokens:
            try:
                v.append(rankDict[j])
            except KeyError:
                continue
        try:
            x.append(median(v))
        except StatisticsError:
            x.append(-1)
    x = pd.DataFrame(x, columns = ["MedianRank"])
        
    df = pd.concat([df, x], axis = 1)
    
    
    #Median
    x = []
    for i in data[targetCol]:
        tokens = stopword(tk(i))
        v = []
        for j in tokens:
            try:
                v.append(rankDict[j])
            except KeyError:
                continue
        try:
            x.append(median(v))
        except StatisticsError:
            x.append(-1)
    x = pd.DataFrame(x, columns = ["MedianContentRank"])
        
    df = pd.concat([df, x], axis = 1)
    
    
    #Min
    x = []
    for i in data[targetCol]:
        tokens = stopword(tk(i))
        v = 999999
        for j in tokens:
            try:
                if rankDict[j] < v:
                    v = rankDict[j]
            except KeyError:
                continue
        x.append(v)
    x = pd.DataFrame(x, columns = ["MinContentRank"])
        
    df = pd.concat([df, x], axis = 1)
    
    df.to_csv(targetFile, index=False)
    
    return pd.concat([data, df], axis = 1)
    

# misc(?)
vad = pd.read_csv("data/misc/BRM-emot-submit.csv", index_col = 1)
freq = pd.read_csv("data/misc/unigram_freq.csv")
