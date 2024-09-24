import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st

# Mapping of word index back to words for understanding
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Cache the model loading to avoid reloading it multiple times
@st.cache_resource
def load_sentiment_model():
    return load_model('Simple_RNN_imdb.h5')

# Load the model
model = load_sentiment_model()

def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter the movie review to classify it as positive or negative.')

user_input = st.text_area('Movie review')

if st.button('Classify'):
    if user_input.strip():  # Check if input is not empty
        sentiment, score = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.4f}')
    else:
        st.write('Please enter a valid movie review.')
