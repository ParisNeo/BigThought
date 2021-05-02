# BigThought
# Author      : Saifeddine ALOUI
# Description : A deep neural network to find the answer to life the universe and everything
# Requirements :
# Please download question answer from :
# https://rajpurkar.github.io/SQuAD-explorer/
# Create a folder called data
# Put the downloaded json file in the folder data
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import itertools
import re
from pathlib import Path

maxlen = 100

data = json.load(open("data/train-v2.0.json","r"))

#Lets extract raw questions and answers
questions =  [data["data"][i]["paragraphs"][j]["qas"][k]["question"].lower() for i in range(len(data["data"])) for j in range(len(data["data"][i]["paragraphs"])) for k in range(len(data["data"][i]["paragraphs"][j]["qas"])) for a in range(len(data["data"][i]["paragraphs"][j]["qas"][k]["answers"]))]
answers =  [data["data"][i]["paragraphs"][j]["qas"][k]["answers"][a]["text"].lower() for i in range(len(data["data"])) for j in range(len(data["data"][i]["paragraphs"])) for k in range(len(data["data"][i]["paragraphs"][j]["qas"])) for a in range(len(data["data"][i]["paragraphs"][j]["qas"][k]["answers"]))]

# Add our question
questions.append("what is the the answer to life the universe and everything?")
answers.append("42")

#re.sub('"','',re.sub("'",'',t))
all = questions + answers
words = list(itertools.chain.from_iterable([re.sub(' +',' ', t.replace('"','').replace("'",'').replace("(",'').replace(")",'').replace("?",'').replace(",",' ')).split(" ") for t in all]))
words = np.unique(np.array(words)).tolist()
print(words)

# Build a vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(words)+1)
tokenizer.fit_on_texts(words)

# pad our sentences to get fixed size sentences

X = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(questions), padding='post', maxlen=maxlen)*(2/(len(words)+1))-1
y = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(answers), padding='post', maxlen=maxlen)*(2/(len(words)+1))-1

#Let's build our model
def resbloc(h0):
    h = tf.keras.layers.Dense(50,activation="relu")(h0)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(50,activation="relu")(h)
    h = tf.keras.layers.Dense(50,activation="relu")(h)
    h = tf.keras.layers.Concatenate()([h0,h])
    return h

question = tf.keras.layers.Input(shape=(maxlen),name="Question")

# input blocs
h = tf.keras.layers.Reshape((maxlen,1))(question)
h = tf.keras.layers.LSTM(100, return_sequences=True)(h)
h = tf.keras.layers.LSTM(50)(h)
h0= tf.keras.layers.Reshape((50,))(h)

# Resblocs
"""
h = resbloc(h0)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
"""
# Get as deep as you want ..

# Final blocs
#h = tf.keras.layers.Dense(50,activation="relu")(h)
# Shortcut
#h = tf.keras.layers.Dense(100,activation="relu")(h)
answer = tf.keras.layers.Dense(maxlen,activation="tanh", name="Answer")(h)

model = tf.keras.models.Model(question, answer)
#find an old model
model_folder = Path("model")
if not model_folder.exists():
    model_folder.mkdir(exist_ok=True, parents=True)

model_path = model_folder/"bigThought.hdf5"

if model_path.exists():
    try:
        model.load_weights(str(model_path))
        print("Weights loaded")
    except:
        print("Incompatible weights found.")


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")


question = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([input("Question :")]), padding='post', maxlen=maxlen)
print(question)
out = model.predict(question*(2/(len(words)+1))-1)
print(out)
answer = np.round((out+1)*(len(words)+1)/2)
print(answer)
txt = tokenizer.sequences_to_texts(answer)
print(txt)


