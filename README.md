![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawBERT%20&fontColor=f7e600&textBg=true&fontSize=100)

<img src="https://img.shields.io/badge/BERT-f7e600?style=flat-square&logo=Gitee&logoColor=black"/> <img src="https://img.shields.io/badge/Python-f7e600?style=flat-square&logo=Python&logoColor=black"/> <img src="https://img.shields.io/badge/Colab-f7e600?style=flat-square&logo=Google Colab&logoColor=black"/> 


## Bidirectional Encoder Representations from Transformers for using Korean Legal Text

## 1. Model description
 **BERT (Bidirectional Encoder Representations from Transformers)** is a pre-trained large language model based on the Transformers encoder. We can use existing various BERT-based large language models. However, this way is less competitive in the field of legal information retrieval. Therefore, we release a ***KRLawBERT*** pre-trained on large-scale legal text dataset by benchmarking two popular techniques: **Masked Language Modeling (MLM)** and **Transformer-based Sequential Denoising Auto-Encoder (TSDAE)**. In particular, to improve performance, KRLawBERT was lexicographically developed by applying **various MLM approaches** as follows.


## 2. Model Usage
### Pre-train
Our model is pre-trained using four masking techniques: **Statistical MLM**, **Dynamic MLM**, **Frequency MLM**, and **TSDAE**.

### 1. Statistical Masked Language Modeling
**Statistical Masked Language Modeling** implemented a pretraining script for training a masked language model (MLM) using the BERT architecture with PyTorch and Hugging Face's Transformers library. This code trains the original MLM on a custom dataset. Overall, this script trains a BERT-based MLM on a custom dataset and saves the model with the best loss during training. You can use this trained MLM model for downstream tasks such as text generation, text classification, or any task that benefits from pre-trained language representations.

```python
$ python pre-training/statistical-MLM.py
```

### 2. Dynamic Masked Language Modeling
**Dynamic Masked Language Modeling** implemented a dynamic masked language model (MLM) pretraining script using the BERT architecture with PyTorch and the Transformers library. This script uses a custom dataset and a dynamic masking strategy similar to RoBERTa. This dynamic MLM training strategy is designed to help the model adapt better to downstream tasks by exposing it to more diverse masked tokens during pretraining. You can choose between the two strategies(statistical or dynamic) based on your specific use case and evaluation results.

```python
$ python pre-training/dynamic-MLM.py
```

### 3. Frequency-based Masked Language Modeling
**Frequency-based Masked Language Modeling** implemented a frequency-based masked language model (MLM) pretraining script using the BERT architecture with PyTorch and the Transformers library. This script uses a custom dataset and an advanced masking strategy based on token frequency. You now have three different MLM pretraining strategies (statistical, dynamic, and frequency-based) that you can choose from based on your specific use case and evaluation results. Each strategy exposes the model to different training data patterns, which may be beneficial for different downstream tasks.

```python
$ python pre-training/frequency-MLM.py
```

### 4. Transformer-based Sequential Denosing Auto-Encoder
**Transformer-based Sequential Denosing Auto-Encoder(TSDAE)** introduces noise to input sequences by deleting or swapping tokens. These damaged sentences are encoded by the transformer model into sentence vectors. Another decoder network then attempts to reconstruct the original input from the damaged sentence encoding. This may seem similar to masked-language modeling (MLM). MLM is the most common pretraining approach for transformer models. A random number of tokens are masked using a **‘masking token’**, and the transformer must try to guess what is missing, like a ‘fill in the blanks’ test in school. TSDAE differs in that the decoder in MLM has access to full-length word embeddings for **every single token**. The **TSDAE** decoder only has access to the **sentence vector** produced by the encoder.
```python
$ python pre-training/TSDAE.py
```

### Fine-tuning

### 1. NIL

### 2. STS


## 3. Dev
  - Seoul National University NLP Labs
  - Navy Lee
