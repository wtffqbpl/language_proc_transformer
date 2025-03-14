#! coding: utf-8

import unittest
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf



class IntegrationTest(unittest.TestCase):

    def test_datasets(self):
        all_datasets = list_datasets()
        print(f'there are {len(all_datasets)} datasets currently available on the Hub')

        print(f'The first 10 are: {all_datasets[:10]}')

        emotions = load_dataset('emotion')
        print(emotions)

        train_ds = emotions['train']
        print(train_ds)

        emotions.set_format(type='pandas')
        df = emotions['train'][:]
        print(df.head())

        # Create a new column in our DataFrame with teh corresponding label names.
        def label_int2str(row):
            return emotions['train'].features['label'].int2str(row)

        df['label_name'] = df['label'].apply(label_int2str)
        print(df.head())

        df['label_name'].value_counts(ascending=True).plot.barh()
        plt.title('Frequency of Classes')
        plt.show()

        self.assertTrue(True)

    def test_auto_tokenization(self):
        text = "tokenizing text is a core task of NLP"

        model_ckpt = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        # Tokenize the text
        encoded_text = tokenizer(text)

        print(encoded_text)

        # We can convert them back the index into tokens by using the tokenizer's convert_ids_to_tokens
        tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
        # The [CLS] and [SEP] tokens have been added to the start and end of the sequence.
        # These tokens differ from model to model, but their main role is to indicate the
        # start and end of a sequence.
        # The ## prefix in ##izing and ##p means that the preceding string is not whitespace;
        # any token with this prefix should be merged with the previous token when you
        # convert the tokens back to a string.
        expected_output = ['[CLS]', 'token', '##izing', 'text', 'is',
                           'a', 'core', 'task', 'of', 'nl', '##p', '[SEP]']
        self.assertEqual(expected_output, tokens)

        # Convert tokens back to string
        converted_str = tokenizer.convert_tokens_to_string(tokens)
        expected_str = "[CLS] tokenizing text is a core task of nlp [SEP]"
        self.assertEqual(expected_str, converted_str)

        # We can inspect the vocabulary size.
        print(f'{tokenizer.vocab_size=}')

        # And the corresponding model's maximum context size.
        print(f'{tokenizer.model_max_length=}')

        # Another interesting attribute to know about is the names of the fields that the
        # model expects in its forward pass:
        print(f'{tokenizer.model_input_names=}')

        # If you want to load the specific class manually you can do as follows.
        tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

        # When using pretrained models, it is really important to make sure
        # that you use the same tokenizer that the model was trained with.
        # From the model's perspective, switching the tokenizer is like shuffling
        # the vocabulary. If everyone around you started swapping random words
        # like 'house' for 'cat', you'd have a hard time understanding what was
        # going on too.

    def test_tokenizing_the_whole_dataset(self):
        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)

        model_ckpt = 'distilbert-base-uncased'
        tokenizer= AutoTokenizer.from_pretrained(model_ckpt)

        print(f'{tokenizer.special_tokens_map=}')

        emotions = load_dataset('emotion')
        print(tokenize(emotions['train'][:2]))

        emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
        # We can apply it across all the splits in the corpus in a single line of code.
        # By default, the map() method operators individually on every example in the corpus,
        # so setting batched=True will encode the tweets in batches. Because we've set
        # batch-size=None, our tokenize() function will be applied on the full dataset as a
        # single batch.
        print(emotions_encoded['train'].column_names)

        self.assertTrue(True)

    def test_auto_model(self):
        model_ckpt = 'distilbert-base-uncased'
        device = dlf.devices()[0]
        model = AutoModel.from_pretrained(model_ckpt).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)

        emotions = load_dataset('emotion')

        emotions_encoded = emotions.map(tokenize, batched=True)

        text = "this is a test"
        inputs = tokenizer(text, return_tensors='pt')
        print(f"Input tensor shape: {inputs['input_ids'].size()}")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        print(outputs)

        print(f'{outputs.last_hidden_state.size()=}')

        self.assertTrue(True)

        def extract_hidden_states(batch) -> dict[str, Any]:
            # place model inputs on the specified device
            inputs_ = {k: v.to(device) for k, v in batch.items()
                      if k in tokenizer.model_input_names}
            # Extract last hidden states
            with torch.no_grad():
                last_hidden_state = model(**inputs_).last_hidden_state
            # Return vector for [CLS] token
            return {'hidden_state': last_hidden_state[:, 0].cpu().numpy()}

        # Convert the input_ids and attention_mask columns to the 'torch' format
        emotions_encoded.set_format('torch',
                                    columns=['input_ids', 'attention_mask', 'label'])
        emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

        print(emotions_hidden['train'].column_names)


if __name__ == '__main__':
    unittest.main(verbosity=True)
