#! pip install transformers
#! pip install -U sentence-transformers
#! pip install sentencepiece

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader, InputExample
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader
import pandas as pd
import math
import logging
device = "cuda" if torch.cuda.is_available() else "cpu"
print(">> Using {}".format(device))

################################
### 1. Load PLMs (KRLawBERT) ###
################################
from models.kor_sentence_bert import *
# 1-1. Load pre-trained or re-trained KoLawBERT
model_name = './models/tsdae_trained_model'    # or 'mlm_trained_model' or 'dynamic_mlm_trained_model' or 'freq_mlm_trained_model'
token_name = 'klue/bert-base'    # if Kor : 'klue/bert-base'   else if Eng : 'bert-base-uncased'
model = make_sentenceBERT(model_name, token_name)

# 1-2. If load re-trained KRLawBERT / TSDAE / Huggingface PLMs
#model_name = "klue/bert-base"
#model = SentenceTransformer(model_name)

########################################
### 2. Fine-Tuning on KorNIL Dataset ###
########################################
# Dataset for Train
train_snli = pd.read_csv("data/snli_1.0_train.ko.tsv", sep='\t', quoting=3)  
train_xnli = pd.read_csv("data/multinli.train.ko.tsv", sep='\t', quoting=3)
train_data = pd.concat([train_snli, train_xnli], ignore_index=True)
print(">> Total Train Dataset size :", len(train_data))

# Dataset for Eval
val_data = pd.read_csv("data/sts-dev.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("data/sts-test.tsv", sep='\t', quoting=3)
print(">> Total Validataion Dataset size :", len(val_data))
print(">> Total Test Dataset size :", len(test_data))

# label_dict
label_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

# 2-1. Make NIL Dataset for Training
# Traing dataset
train_batch_size = 16
train_samples = make_kornli_dataset(train_data)
# Train DataLoader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# Val/Test dataset
val_data = drop_korsts(val_data)
test_data = drop_korsts(test_data)
dev_samples = make_korsts_dataset(val_data)
test_samples = make_korsts_dataset(test_data)
# Eval DataLoader
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Loss function : Calculate MSE loss
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=len(label_dict))

# 2-2. Training
# Warmup(10% of train data for warm-up) & Epochs
num_epochs = 3
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)   
logging.info("Warmup-steps: {}".format(warmup_steps))

model_save_path = 'models/nil_krlawbert'

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

# 2-3. Evaluation
#model_save_path = 'output/nil_task_bert/0_Transformer'
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
print(">> Best TEST Score is : {:.4f}".format(test_evaluator(model, output_path=model_save_path)))


########################################
### 3. Fine-Tuning on KorSTS Dataset ###
########################################
model_name = 'models/nil_krlawbert'
model_save_path = 'models/krlawbert'
model = SentenceTransformer(model_name)

# 3-1. Make STS Dataset for Training
train_batch_size = 16
train_data = pd.read_csv("data/sts-train.tsv", sep='\t', quoting=3)
val_data = pd.read_csv("data/sts-dev.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("data/sts-test.tsv", sep='\t', quoting=3)

train_data = drop_korsts(train_data)
val_data = drop_korsts(val_data)
test_data = drop_korsts(test_data)

# Traing/val/test dataset
train_samples = make_korsts_dataset(train_data)
dev_samples = make_korsts_dataset(val_data)
test_samples = make_korsts_dataset(test_data)

# DataLoader
train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

# Loss function : Calculate Cosine similarity
train_loss = losses.CosineSimilarityLoss(model=model)

# Evaluator 
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# 3-2. Training
# Warmup(10%) & Epochs
num_epochs = 15
warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

# 3-3. Evaluation
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
print(">> Best TEST Socre is : {:.4f}".format(test_evaluator(model, output_path=model_save_path)))
