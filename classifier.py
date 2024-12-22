#! coding: utf-8

import torch
from transformers import DistilBertTokenizer, DistilBertModel, AutoModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ckpt = 'distilbert-base-uncased'
# Initialize both tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

# from_pretrained() method loads the weights of a pretrained model.
# The AutoModel class converts the token encodings to embeddings, and then feeds
# them through the encoder stack to return the hidden stats.
model = AutoModel.from_pretrained(model_ckpt).to(device)

# First tokenize the text
text = "this is a test"
# inputs = tokenizer(text, return_tensors='pt')

# When tokenizing, the tensors can be moved to device after tokenization
inputs = tokenizer(text, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}

# Then pass the tokenized inputs to the model
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(f"Input tensor shape: {inputs['input_ids'].shape}")

