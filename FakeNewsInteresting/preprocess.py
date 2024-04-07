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