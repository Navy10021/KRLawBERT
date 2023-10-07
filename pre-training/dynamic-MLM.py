# *** Advanced style dynamic masking *** #

#! pip install nltk
#! pip install transformers
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize     # sentence tokenizer
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer
import matplotlib.pyplot as plt


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

# 3. Dynamic MLM
class DynamicMLMPretrainingDataset(Dataset):
    def __init__(self, text_data, tokenizer):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_length = self.find_max_len()

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]

        # Tokenize the text and Padding
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens + [tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # Create masked input and labels for MLM
        masked_tokens, labels = self.mask_tokens(tokens)

        return torch.tensor(masked_tokens), torch.tensor(labels)

    def find_max_len(self):
        # Find max length
        tokenized_text_data = [self.tokenizer.encode(text, add_special_tokens=True) for text in self.text_data]
        max_length = max(len(tokens) for tokens in tokenized_text_data)
        print("Text max length : {}".format(max_length))

        return max_length

    def mask_tokens(self, tokens):
        # RoBERTa-style dynamic masking
        masked_indices = torch.rand(len(tokens)) < 0.15  # 15% chance of randomly masking (same as RoBERTa)
        masked_tokens = torch.tensor(tokens)
        masked_tokens[masked_indices] = self.tokenizer.mask_token_id
        labels = torch.tensor(tokens)
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        return masked_tokens, labels


# 4. Create a DataLoader for batch training
dataset = DynamicMLMPretrainingDataset(text_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 5. Loss function and optimizer (you should fine-tune BERT's pre-trained weights)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 6. Training loop
epochs = 10
best_loss = float('inf')  # Initialize the best_loss with positive infinity
loss_values_2 = list()
model.train()
for epoch in range(epochs):
    for batch_masked_tokens, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_masked_tokens, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    loss_values_2.append(loss.item())
    print(f"Epoch {epoch + 1}/{epochs} - Loss : {loss.item()}")

    # 7. Save the MLM trained model for later use when a new best loss is achieved
    if loss < best_loss:
        best_loss = loss
        model.save_pretrained('./models/dynamic_mlm_trained_model')


# 8. Print loss graph
x = [i for i in range(0, len(loss_values_2))]
y = loss_values_2
# Create a line plot for loss
plt.plot(x, y, marker='s', linestyle='--', color='green')
# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph Over Epochs')
# Display the plot
plt.grid(True)
plt.show()
