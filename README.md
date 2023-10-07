![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawBERT%20&fontColor=f7e600&textBg=true&fontSize=100)

<img src="https://img.shields.io/badge/BERT-f7e600?style=flat-square&logo=Gitee&logoColor=black"/> <img src="https://img.shields.io/badge/Python-f7e600?style=flat-square&logo=Python&logoColor=black"/> <img src="https://img.shields.io/badge/Colab-f7e600?style=flat-square&logo=Google Colab&logoColor=black"/> 


## Bidirectional Encoder Representations from Transformers for using Korean Legal Text

## 1. Model description
 **BERT (Bidirectional Encoder Representations from Transformers)** is a pre-trained large language model based on the Transformers encoder. We can use existing various BERT-based large language models. However, this way is less competitive in the field of legal information retrieval. Therefore, we release a ***KRLawBERT*** pre-trained on large-scale legal text dataset by benchmarking two popular techniques: **Masked Language Modeling (MLM)** and **Transformer-based Sequential Denoising Auto-Encoder (TSDAE)**. In particular, to improve performance, KRLawBERT was lexicographically developed by applying various MLM approaches as follows.


## 2. Model Usage
### Pre-train
Our model is pre-trained using four masking techniques: Statistical MLM, Dynamic MLM, Frequency MLM, and TSDAE.

### 1. Statistical Masked Language Modeling

### 2. Dynamic Masked Language Modeling

### 3. Frequency-based Masked Language Modeling

### 4. Transformer-based Sequential Denosing Auto-Encoder

### Fine-tuning

### 1. NIL

### 2. STS


## 3. Dev
  - Seoul National University NLP Labs
  - Navy Lee
