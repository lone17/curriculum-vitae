"""
We're going to use the wikitext (link) dataset with the distilbert-base-cased (link) 
model checkpoint.

Start by loading the wikitext-2-raw-v1 version of that dataset, and take the 11th 
example (index 10) of the train split.
We'll tokenize this using the appropriate tokenizer, and we'll mask the sixth token 
(index 5) the sequence.

When using the distilbert-base-cased checkpoint to unmask that (sixth token, index 5) 
token, what is the most probable predicted token (please provide the decoded token, and 
not the ID)?

Tips:
- You might find the transformers docs (link) useful.
- You might find the datasets docs (link) useful.
- You might also be interested in the Hugging Face course (link).
"""

import torch
from datasets import load_dataset
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-cased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

sample_index = 10
mask_token_index = 5

input_text = data[sample_index]["text"]

inputs = tokenizer(input_text, return_tensors="pt")
inputs["input_ids"][0][mask_token_index] = tokenizer.mask_token_id

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

print("Predicted token:", tokenizer.decode([predicted_token_id]))

# combat
# mechanic
