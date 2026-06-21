# Development Notes

## Project

Micro GPT – A miniature GPT-style language model built to understand the workflow of Large Language Models using PyTorch.

---

## Dataset

Dataset:
Multi Turn Chatbot Conversation Dataset

Source:
https://www.kaggle.com/datasets/abhayayare/multi-turn-chatbot-conversation-dataset

Reason:
Selected because it contains conversational data suitable for GPT-style next-token prediction.

Rows Used:
50000

Vocabulary Size:
235

Total Encoded Tokens:
246393

---

## Data Processing

Completed:

* Dataset Selection
* Dataset Inspection
* Text Cleaning
* Lowercase Conversion
* Tokenization
* Vocabulary Creation
* Word to Index Mapping
* Index to Word Mapping
* Encoded Token Generation

Generated Files:

* word_to_idx.json
* idx_to_word.json
* encoded_tokens.txt

---

## Positional Encoding

Implemented sinusoidal positional encoding using NumPy.

Generated:

* positional_encoding.py
* positional_encoding.png

Purpose:

Provide positional information to token embeddings before entering the transformer architecture.

---

## Transformer Model

Configuration:

* Embedding Dimension: 64
* Attention Heads: 2
* Transformer Layers: 1
* Sequence Length: 5

Framework:

* PyTorch

Model Components:

* Embedding Layer
* Transformer Encoder
* Multi-Head Attention (PyTorch implementation)
* Feed Forward Network
* Output Linear Layer

Total Parameters:

311,467

---

## Training

Loss Function:

* CrossEntropyLoss

Optimizer:

* Adam Optimizer

Training Subset:

10000 samples

Training Results:

Epoch 1 Loss: 5.7854

Epoch 2 Loss: 4.9621

Epoch 3 Loss: 4.3164

Observation:

Loss decreased consistently, indicating successful learning.

---

## Text Generation

Implemented:

* Greedy Decoding
* Top-k Sampling
* Beam Search

Sample Outputs:

Greedy:
best deals today sure let me help you with that help you with that help

Top-k:
best deals today sure let believe money online seo with believe an speaking is fit

Beam Search:
best deals today sure let me help you with that

---

## Model Persistence

Saved Model:

* micro_gpt_model.pth

Purpose:

Allows loading the trained model without retraining.

---

## CLI Tool

Implemented:

* cli.py

Features:

* Accepts user prompt
* Loads trained model
* Generates next-word prediction

Example:

Input:
Hi how is your day today

Output:
sure

---

## Project Status

Completed Successfully

All objectives defined in the internship project PDF were implemented and tested.

