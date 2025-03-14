#! coding: utf-8

import unittest
import torch
from transformers import pipeline
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.dlf as dlf


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = dlf.devices()[0]
        self.text = """
        Dear Amazon, last week I ordered an Optimus Prime action figure
        from your online store in Germany. Unfortunately, when I opened the package,
        I discovered to my horror that I had been sent an action figure of Megatron
        instead! As a lifelong enemy of the Decepticons, I hope you can understand my
        dilemma. To resolve the issue, I demand an exchange of megatron for the
        Optimus Prime figure I ordered. Enclosed are copies of my records concerning
        this purchase. I expect to hear from you soon. Sincerely, Bumblebee.
        """
        pass

    def test_text_generation(self):
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        generator = pipeline('text-generation', model='gpt2', tokenizer=tokenizer, device=self.device)
        generator.pad_token = tokenizer.eos_token
        response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
        prompt = self.text + '\n\nCustomer service response:\n' + response
        output = generator(prompt, max_length=256, truncation=True, pad_token_id=tokenizer.eos_token_id)
        print(output[0]['generated_text'])


if __name__ == '__main__':
    unittest.main(verbosity=True)
