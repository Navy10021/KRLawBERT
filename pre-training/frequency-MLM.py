# *** Advanced dynamic masking(Tokens frequency based) *** #
# implements an advanced masking strategy based on token frequencies. We count the frequency of each token in the input text and sort tokens by frequency.
# Then, we mask a portion (e.g., bottom 10%) of the less frequent tokens.

#! pip install nltk
#! pip install transformers
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize     # sentence tokenizer
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer


# 1. Read text file and sentence tokenization  
def read_and_tokenize(file_path):
    sentences = list()
    with open(file_path, 'r', encoding = 'utf-8') as file:
        text = file.read()
        sentences.extend(sent_tokenize(text))
    return sentences

file_path = 'data/US_law_cases.txt'
text_data = read_and_tokenize(file_path)

# print 5 sentences
for i, sent in enumerate(text_data[:5]):
    print(sent)

# 2. Initialize a BERT tokenizer and model 
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer

# 2. Initialize a BERT tokenizer and model 
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# 3. Statistical MLM
class FrequencyMLMPretrainingDataset(Dataset):
    def __init__(self, text_data, tokenizer):
        self.text_data = text_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        
        # Find max length
        tokenized_text_data = [self.tokenizer.encode(text, add_special_tokens=True) for text in self.text_data]
        max_length = max(len(tokens) for tokens in tokenized_text_data)
        #print("Text max length : {}".format(max_length))

        # Tokenize the text and Padding
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
        
        # Create masked input and labels for MLM
        masked_tokens, labels = self.mask_tokens(tokens)
        
        return torch.tensor(masked_tokens), torch.tensor(labels)

    def mask_tokens(self, tokens):
        # Advanced masking strategy: Mask tokens based on token frequency
        masked_tokens = torch.tensor(tokens)
        labels = torch.tensor(tokens)
        
        # Determine token frequencies in the input
        token_counts = {}
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
        
        # Sort tokens by frequency (less frequent tokens first)
        sorted_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x])

        # Mask a portion of less frequent tokens (e.g., bottom 15%)
        mask_percentage = 0.15
        num_tokens_to_mask = int(len(sorted_tokens) * mask_percentage)

        for i in range(num_tokens_to_mask):
            token_to_mask = sorted_tokens[i]
            masked_tokens[masked_tokens == token_to_mask] = self.tokenizer.mask_token_id
            labels[labels == token_to_mask] = -100  # Only compute loss on masked tokens
        
        return masked_tokens, labels

# 4. Create a DataLoader for batch training
dataset = FrequencyMLMPretrainingDataset(text_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 5. Loss function and optimizer (you should fine-tune BERT's pre-trained weights)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 6. Training loop
epochs = 10
best_loss = float('inf')  # Initialize the best_loss with positive infinity
model.train()
for epoch in range(epochs):
    for batch_masked_tokens, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_masked_tokens, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{epochs} - Loss : {loss.item()}")

    if loss < best_loss:
        best_loss = loss
        # 7. Save the MLM trained model for later use when a new best loss is achieved
        model.save_pretrained('./models/freq_mlm_trained_model')
