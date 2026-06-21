# Micro GPT

## Overview

Micro GPT is a miniature GPT-style language model developed as part of a Machine Learning Internship project. The objective was to understand the complete workflow behind modern Large Language Models (LLMs), including dataset preparation, tokenization, positional encoding, transformer architecture, model training and text generation.

This project was built for educational and research purposes to gain hands-on experience with transformer-based language models using PyTorch.

---

## Internship Context

This project was developed during Machine Learning internship focused on:

* Natural Language Processing (NLP)
* Transformer Architecture
* GPT-style Language Models
* Tokenization and Encoding
* Model Training and Inference

---

## Learning Objectives

Before starting this project, the focus was on understanding the theory behind:

* Large Language Models
* Tokenization
* Attention Mechanisms
* Transformers
* Text Generation

After completing the project, I gained practical experience in:

* Data preprocessing
* Vocabulary creation
* Positional encoding
* Transformer implementation using PyTorch
* Model training
* Greedy decoding
* Top-k sampling
* Beam search
* Building a command-line inference interface

---

## Dataset

Dataset Name:

Multi Turn Chatbot Conversation Dataset

Source:

https://www.kaggle.com/datasets/abhayayare/multi-turn-chatbot-conversation-dataset

Dataset Usage:

* 50,000 rows used for experimentation
* Conversational text suitable for next-token prediction

Statistics:

* Vocabulary Size: 235
* Total Encoded Tokens: 246,393

---

## Features

* Dataset Preparation
* Text Cleaning
* Tokenization
* Vocabulary Creation
* Word-to-Index Mapping
* Index-to-Word Mapping
* Encoded Token Generation
* Positional Encoding
* Transformer Encoder Architecture
* Multi-Head Attention
* Model Training using PyTorch
* Greedy Decoding
* Top-k Sampling
* Beam Search
* Command Line Interface (CLI)

---

## Model Configuration

| Parameter           | Value   |
| ------------------- | ------- |
| Embedding Dimension | 64      |
| Attention Heads     | 2       |
| Transformer Layers  | 1       |
| Sequence Length     | 5       |
| Vocabulary Size     | 235     |
| Total Parameters    | 311,467 |

---

## Training Results

| Epoch | Loss   |
| ----- | ------ |
| 1     | 5.7854 |
| 2     | 4.9621 |
| 3     | 4.3164 |

The decreasing loss indicates that the model learned successfully during training.

---

## Sample Outputs

### Greedy Decoding

best deals today sure let me help you with that help you with that help

### Top-k Sampling

best deals today sure let believe money online seo with believe an speaking is fit

### Beam Search

best deals today sure let me help you with that

---

## Project Structure

```text
micro-gpt/
├── cli.py
├── tokenizer.py
├── positional_encoding.py
├── models/
│   └── micro_gpt_model.pth
├── data/
│   └── encoded_tokens.txt
├── word_to_idx.json
├── idx_to_word.json
├── README.md
└── Notes.md
```

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Google Colab

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ihsan-03/micro-gpt.git
cd micro-gpt
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch pandas numpy matplotlib
```

---

## Running the CLI

Execute:

```bash
python cli.py
```

Example:

```text
Enter Prompt: Hi how is your day today

Generated Word: sure
```

---

## Challenges Faced

During development several challenges were encountered:

* Large dataset processing
* Vocabulary generation and token encoding
* Understanding transformer architecture
* Implementing different decoding strategies
* Model persistence and loading
* Model training and inference

These challenges provided practical exposure development workflow.

---

## Future Improvements

* Larger dataset training
* Better tokenizer
* Longer text generation
* OpenWebUI integration
* Local chat interface
* Improved transformer architecture

## Project Status

Completed Successfully as part of a Machine Learning Internship project.

