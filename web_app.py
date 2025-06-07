import streamlit as st
from predict import predict_sentiment

st.title("Sentiment Analysis")
user_input = st.text_area("Enter a review:")
if st.button("Predict Sentiment"):
    result = predict_sentiment(user_input)
    st.write(f"Sentiment: **{result}**")
