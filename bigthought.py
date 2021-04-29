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

data = json.load(open("data/train-v2.0.json","r"))

#Lets extract raw questions and answers
questions = [data["data"][i]["paragraphs"][j]["qas"][k]["question"] for i in range(len(data["data"])) for j in range(len(data["data"][i]["paragraphs"])) for k in range(len(data["data"][i]["paragraphs"][j]["qas"])) for a in range(len(data["data"][i]["paragraphs"][j]["qas"][k]["answers"]))]
answers = [data["data"][i]["paragraphs"][j]["qas"][k]["answers"][a]["text"] for i in range(len(data["data"])) for j in range(len(data["data"][i]["paragraphs"])) for k in range(len(data["data"][i]["paragraphs"][j]["qas"])) for a in range(len(data["data"][i]["paragraphs"][j]["qas"][k]["answers"]))]

# Add our question
questions.append("what is the the answer to life the universe and everything?")
answers.append("42")

#re.sub('"','',re.sub("'",'',t))
all = questions + answers
words = list(itertools.chain.from_iterable([re.sub(' +',' ', t.replace('"','').replace("'",'').replace("(",'').replace(")",'')).split(" ") for t in all]))
words = np.unique(np.array(words)).tolist()
print(words)

# Build a vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(words)+1)
tokenizer.fit_on_texts(words)

# pad our sentences to get fixed size sentences
maxlen = 100
X = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(questions), padding='post', maxlen=maxlen)
y = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(answers), padding='post', maxlen=maxlen)

#Let's build our model
def resbloc(h0):
    h = tf.keras.layers.Dense(100,activation="relu")(h0)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(50,activation="relu")(h)
    h = tf.keras.layers.Dense(100,activation="relu")(h)
    h = tf.keras.layers.Concatenate()([h0,h])
    return h

question = tf.keras.layers.Input(shape=(100,),name="Question")

# input blocs
h = tf.keras.layers.Dense(1000,activation="relu")(question)
# h = tf.keras.layers.Reshape((1000,1))(h)
# h = tf.keras.layers.Conv1D(128, 5, activation='tanh')(h)
#h = tf.keras.layers.GlobalMaxPooling1D()(h)
h0 = tf.keras.layers.Dense(100,activation="relu")(h)

# Resblocs
h = resbloc(h0)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
h = resbloc(h)
# Get as deep as you want ..

# Final blocs
h = tf.keras.layers.Dense(100,activation="relu")(h)
# Shortcut
h = tf.keras.layers.Concatenate()([h,h0])
h = tf.keras.layers.Dropout(0.25)(h)
h = tf.keras.layers.Dense(1000,activation="relu")(h)
answer = tf.keras.layers.Dense(100,activation="relu", name="Answer")(h)

model = tf.keras.models.Model(question, answer)
#find an old model
model_folder = Path("model")
if not model_folder.exists():
    model_folder.mkdir(exist_ok=True, parents=True)

model_path = model_folder/"bigThought.hdf5"

if model_path.exists():
    model.load_weights(str(model_path))
    print("Weights loaded")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")

# Now let it learn for a long long long time it learns very slowly
model.fit(X,y, batch_size=256, epochs=15)

#Don't forget to save it
model.save_weights(str(model_path))

# Now it is our time to ask the ultimate question !
question = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(["what is the the answer to life the universe and everything?"]), padding='post', maxlen=maxlen)
print(question)
print(model.predict(question))
answer = (model.predict(question)).astype(int)
print(answer)
txt = tokenizer.sequences_to_texts(answer)
print(txt)







