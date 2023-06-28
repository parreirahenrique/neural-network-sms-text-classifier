import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.request import Request, urlopen

# Get project files
if not os.path.exists("train-data.tsv"):
    url = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
    req = Request(
        url=url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )

    webpage = urlopen(req)

    with open("train-data.tsv","wb") as output:
        output.write(webpage.read())
    
    url = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"
    req = Request(
        url=url, 
        headers={"User-Agent": "Mozilla/5.0"}
    )

    webpage = urlopen(req)

    with open("valid-data.tsv","wb") as output:
        output.write(webpage.read())

# Create train and test datasets
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset = pd.read_csv(train_file_path, sep="\t", )
train_dataset.columns = ["type", "message"]

test_dataset = pd.read_csv(test_file_path, sep="\t", )
test_dataset.columns = ["type", "message"]

full_dataset = pd.concat([train_dataset, test_dataset]).reset_index(drop=True)

train_dataset["type"].replace(
    ["ham", "spam"],
    [0, 1],
    inplace=True
)

test_dataset["type"].replace(
    ["ham", "spam"],
    [0, 1],
    inplace=True
)

train_dataset.head()


# Separate the train and test labels
train_labels = train_dataset.pop("type")
test_labels = test_dataset.pop("type")

# Create train and test labels arrays
train_labels_array = np.array(train_labels)
test_labels_array = np.array(test_labels)

# Count number of unique words
message_list = full_dataset["message"].tolist()
message_string = ""

for message in message_list:
    message_string += message + " "

word_list = message_string.split(" ")
word_set = set(word_list)
vocab_size = len(word_set)

# Count maximum number of words in a message
max_length = -1

for message in message_list:
    if len(message.split(" ")) > max_length:
        max_length = len(message.split(" "))

print(f"Maximum number of words in a message: {max_length}")

# Tokenize dataset
embedding_dim = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(full_dataset['message'].tolist())
word_index = tokenizer.word_index

# Get the sequences of tokens for each message in the dataset
train_sequences = tokenizer.texts_to_sequences(train_dataset['message'].tolist())
test_sequences = tokenizer.texts_to_sequences(test_dataset['message'].tolist())

# Pad the sequences
train_padded = pad_sequences(
    train_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

test_padded = pad_sequences(
    test_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train the model
num_epochs = 100

history = model.fit(
    train_padded,
    train_labels_array,
    epochs=num_epochs,
    validation_data=(test_padded, test_labels_array)
)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text: str):
    text_token = tokenizer.texts_to_sequences(pred_text.split(" "))

    flat_text_token = []

    for token in text_token:
        flat_text_token.append(token[0])

    text_padded = pad_sequences(
        [flat_text_token],
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type
    )
    
    [[pred]] = model.predict(text_padded)

    if pred < 0.5:
        prediction = [0, "ham"]
    else:
        prediction = [1, "spam"]
      
    return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)

    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()