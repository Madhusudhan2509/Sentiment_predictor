# Sentiment Analysis using LSTM

This project builds an LSTM-based model to classify text reviews as **Positive** or **Negative**. It includes a Streamlit web app for real-time sentiment prediction.

---

## Files
- model_training.py — Train the LSTM model
- predict.py — Predict sentiment for input text
- web_app.py — Streamlit app for user interface
- model.h5 & `tokenizer.pickle` — Saved model and tokenizer after training

---

## How to Run

1.Install dependencies:
	```bash
   	pip install tensorflow streamlit numpy pandas nltk

2.Train the model:
	python model_training.py

3.Run the predict.py file
	 python predict.py

4.Run the web app:
	streamlit run web_app.py