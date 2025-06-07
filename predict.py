import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import numpy as np

# Load model and word index
model = load_model("sentiment_model.h5")
word_index = imdb.get_word_index()

def encode_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 is 'unknown'
    return pad_sequences([encoded], maxlen=200)

def predict_sentiment(text):
    encoded_text = encode_text(text)
    prediction = model.predict(encoded_text)[0][0]
    return "Negative" if prediction > 0.5 else "Positive"

# Example usage
text = input("Enter a review: ")
print("Sentiment:", predict_sentiment(text))


