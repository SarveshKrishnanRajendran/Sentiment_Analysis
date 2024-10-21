#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
sys.path.append('/home/appuser/.local/lib/python3.10/site-packages')
import os
os.system('pip install transformers')


import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


model_path = './saved_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

def classify_tweet(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    return "Positive" if torch.argmax(probs) == 1 else "Negative"


st.title("Sentiment Analyzer")
user_input = st.text_area("Type to analyze:", "Enter here...")
if st.button("Analyze"):
    result = classify_tweet(user_input)
    st.write("Sentiment: ", result)


# In[ ]:




