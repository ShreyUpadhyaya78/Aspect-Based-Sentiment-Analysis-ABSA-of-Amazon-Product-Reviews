#!/usr/bin/env python
# coding: utf-8

# In[15]:


from model.bert import bert_ATE, bert_ABSA
from data.dataset import dataset_ATM, dataset_ABSA


# In[16]:


from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[17]:


def load_model(model, path):
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')), strict=False)
    return model


# In[18]:


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)


# In[ ]:





# In[19]:


def predict_model_ABSA(sentence, aspect, tokenizer):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor)
        _, predictions = torch.max(outputs, dim=1)
    
    return word_pieces, predictions, outputs

def predict_model_ATE(sentence, tokenizer):
    word_pieces = []
    tokens = tokenizer.tokenize(sentence)
    word_pieces += tokens

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs

def ATE_ABSA(text):
    terms = []
    word = ""
    x, y, z = predict_model_ATE(text, tokenizer)
    for i in range(len(y)):
        if y[i] == 1:
            if len(word) != 0:
                terms.append(word.replace(" ##",""))
            word = x[i]
        if y[i] == 2:
            word += (" " + x[i])
            
    
    if len(word) != 0:
            terms.append(word.replace(" ##",""))
            

    dict = {}
   
    sentiment = ["Negative","Neutral","Positive"]
    
    if len(terms) != 0:
        for i in terms:
            _, c, p = predict_model_ABSA(text, i, tokenizer)
            dict[i] = sentiment[int(c)]
    keys = list(dict.keys())
    values = list(dict.values())

    for i in range(len(keys)):
        if '##' in keys[i]:
          keys[i-1] = keys[i-1]+keys[i].replace('##','')
          values[i] = None
          keys[i] = None
        
    keys = [item for item in keys if item is not None]
    values = [item for item in values if item is not None]
    output_dict = {"Aspects": keys,"Sentiment":values}
    return output_dict

# In[20]:


model_ABSA = load_model(model_ABSA, 'bert_ABSA2.pkl')
model_ATE = load_model(model_ATE, 'bert_ATE.pkl')



import streamlit as st
import pandas as pd


st.title("Aspect-Based Sentiment Analysis")

# Create a text input widget
text = st.text_area("Enter a review text:")

if st.button("Analyze Sentiment"):
    if text:
        results = ATE_ABSA(text)
        st.table(results)
    else:
        st.warning("Please enter some text for analysis.")
        
if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)





