![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawBERT%20&fontColor=f7e600&textBg=true&fontSize=100)

<img src="https://img.shields.io/badge/BERT-f7e600?style=flat-square&logo=Gitee&logoColor=black"/> <img src="https://img.shields.io/badge/Python-f7e600?style=flat-square&logo=Python&logoColor=black"/> <img src="https://img.shields.io/badge/Colab-f7e600?style=flat-square&logo=Google Colab&logoColor=black"/> 


## KRLawBERT : A Tailored BERT Model for Korean Legal Texts

## Abstract :
In this work, we presents the development and utilization of ***KRLawBERT***, a specialized variant of **BERT (Bidirectional Encoder Representations from Transformers)**, designed for the intricate domain of Korean legal texts. ***KRLawBERT*** is pre-trained on a large-scale legal text dataset, employing innovative techniques such as **Masked Language Modeling (MLM)** and **Transformer-based Sequential Denoising Auto-Encoder (TSDAE)** to enhance its performance in legal information retrieval. The research provides a comprehensive guide on the pre-training of ***KRLawBERT*** using various masking strategies and its subsequent fine-tuning on datasets tailored to legal semantics.

## 1. Model description

### 1.1. Introduction to KRLawBERT
***KRLawBERT*** is introduced as a specialized BERT model tailored for Korean legal texts. It is pre-trained on a large-scale legal text dataset, and its development includes the application of advanced techniques such as MLM and TSDAE to improve its competitive edge in the field of legal information retrieval.

### 1.2. Lexicographical Development of KRLawBERT
 Our research delves into the lexicographical development of ***KRLawBERT***, detailing the application of various MLM approaches to enhance its performance. Four masking techniques—Statistical MLM, Dynamic MLM, Frequency MLM, and TSDAE—are employed to adapt the model to the nuances of the Korean legal language.

## 2. Model Usage
### Pre-training KRLawBERT on Specific Text Data
The pre-training of ***KRLawBERT*** is explicated through the implementation of four masking strategies: Statistical MLM, Dynamic MLM, Frequency MLM, and TSDAE. Each approach exposes the model to different training data patterns, catering to diverse downstream tasks. Users are provided with a step-by-step guide for executing these pre-training strategies.

#### 1. Statistical Masked Language Modeling
**Statistical Masked Language Modeling** implemented a pretraining script for training a **masked language model (MLM) using the BERT architecture** with PyTorch and Hugging Face's Transformers library. This code trains the original MLM on a custom dataset. Overall, this script trains a BERT-based MLM on a custom dataset and saves the model with the best loss during training. Users can set the masking ratio. According to the BERT paper, performance is excellent at a ratio of 15%, and experimental results of our model show good learning ability of the model between **15-20%**. You can use this trained MLM model for downstream tasks such as text classification, text search, or any task that benefits from pre-trained language representations. 

```python
$ python pre-training/statistical-MLM.py
```

#### 2. Dynamic Masked Language Modeling
**Dynamic Masked Language Modeling** implemented a dynamic masked language model (MLM) pretraining script using the BERT architecture with PyTorch and the Transformers library. This script uses a custom dataset and **a dynamic masking strategy similar to RoBERTa**. This dynamic MLM training strategy is designed to help the model adapt better to downstream tasks by exposing it to more diverse masked tokens during pretraining. Dynamic MLM also allows users to set the masking percentage within the code we provide.

```python
$ python pre-training/dynamic-MLM.py
```

#### 3. Frequency-based Masked Language Modeling
**Frequency-based Masked Language Modeling** implemented a frequency-based masked language model (MLM) pre-training script using the BERT architecture with PyTorch and the Transformers library. This script uses a custom dataset and **an advanced masking strategy based on token frequency**. Frequency MLM is designed to increase the concentration of training by increasing the MLM learning rate for tokens with high frequency, and the user can set the rate.

```python
$ python pre-training/frequency-MLM.py
```

#### 4. Transformer-based Sequential Denosing Auto-Encoder
**Transformer-based Sequential Denosing Auto-Encoder(TSDAE)** introduces noise to input sequences by deleting or swapping tokens. These damaged sentences are encoded by the transformer model into sentence vectors. Another decoder network then attempts to reconstruct the original input from the damaged sentence encoding. This may seem similar to masked-language modeling (MLM). MLM is the most common pretraining approach for transformer models. **TSDAE** differs in that the decoder in **MLM** has access to full-length word embeddings for **every single token**. The **TSDAE** decoder only has access to the **sentence vector** produced by the encoder.
```python
$ python pre-training/TSDAE.py
```

You now have three different MLM(statistical, dynamic, and frequency-based) and TSDAE pre-training strategies that you can choose from based on your specific use case and evaluation results. Each strategy exposes the model to different training data patterns, which may be beneficial for different downstream tasks. The experimental results show that among MLM methods, the frequency MLM we designed is the most stable learning method. The TSDAE method is also a good option for BERT pre-training.
```python
$ python pre-training/train_loss_graph.py
```

<p align="center"><img src="./pre-training/train_loss_graph.png" width="65%" height="50%"></p>


### Fine-tuning KRLawBERT for Legal Information Retrieval

 To adapt ***KRLawBERT*** for legal information retrieval, the model undergoes a supervised fine-tuning process on three distinct datasets: **Natural Language Inference (NLI)** pairs, **Semantic Textual Similarity (STS)**, and parallel legal data. This fine-tuning approach ensures that ***KRLawBERT*** produces semantic legal embeddings tailored to the specific requirements of the legal domain.
```python
$ python fine-tuning/fine_tuning.py
```

## 3. Conclusion
In summary, this work contributes a specialized **BERT** model, ***KRLawBERT***, designed to excel in the domain of Korean legal texts. Through advanced pre-training and fine-tuning strategies, ***KRLawBERT*** aims to elevate the accuracy and effectiveness of legal information retrieval systems.

## 4. Development
- Seoul National University NLP Labs
- Under the guidance of Navy Lee
