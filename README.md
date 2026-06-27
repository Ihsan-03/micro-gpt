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
* FastAPI REST API
* OpenAI-Compatible API
* OpenWebUI Integration
* Modular Inference Pipeline

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
├── app.py
├── cli.py
├── cli_greedy.py
├── inference.py
├── tokenizer.py
├── positional_encoding.py
├── models/
│   └── micro_gpt_model.pth
├── data/
│   └── encoded_tokens.txt
├── outputs/
├── word_to_idx.json
├── idx_to_word.json
├── requirements.txt
├── README.md
└── Notes.md
```

---

## Architecture

The following diagram shows the complete workflow of the project, from preparing the dataset to interacting with the model through OpenWebUI.

                 Dataset
                     │
                     ▼
           Data Preprocessing
                     │
                     ▼
              Tokenization
                     │
                     ▼
          Vocabulary Creation
                     │
                     ▼
         Transformer Model (PyTorch)
                     │
       
              ▼
       Trained Model (micro_gpt_model.pth)
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
     cli.py                 inference.py
                                  │
                                  ▼
                              FastAPI Backend
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
         /generate API                  OpenAI-Compatible API
                                             │
                                             ▼
                                        OpenWebUI

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* FastAPI
* Uvicorn
* OpenWebUI
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
pip install -r requirements.txt
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

## Running the FastAPI Server

Start the API server:

```bash
python -m uvicorn app:app --reload
```

Once the server starts, it will be available at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

The project exposes the following REST API endpoints:

| Method | Endpoint               | Description                                |
| ------ | ---------------------- | ------------------------------------------ |
| GET    | `/`                    | Check if the API server is running         |
| POST   | `/generate`            | Generate text from a prompt                |
| GET    | `/v1/models`           | List available models                      |
| POST   | `/v1/chat/completions` | OpenAI-compatible chat completion endpoint |

---

## OpenWebUI Integration

The project can be connected to OpenWebUI using its OpenAI-compatible API.

Configuration:

**Base URL**

```text
http://127.0.0.1:8000/v1
```

**API Key**

Since this project runs locally and does not implement authentication, any API key can be entered when configuring OpenWebUI.

After adding the connection in OpenWebUI, the **micro-gpt** model will appear in the model list and can be used through the web interface.

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

## Additional Enhancements

After completing the core language model, I continued improving the project by adding a few new features and making the code easier to use and extend.

* Moved the inference code into a separate inference.py file so it can be reused by different applications.
* Built a FastAPI backend to interact with the model through REST APIs.
* Added a simple text generation API (/generate).
* Implemented OpenAI-compatible endpoints (/v1/models and /v1/chat/completions) so the model can work with OpenAI-supported tools.
* Connected the model to OpenWebUI, allowing it to be used through a web-based chat interface.
* Improved text generation by replacing greedy decoding with Top-k Sampling for more varied responses.


## Future Improvements

* Train the model on a larger dataset.
* Improve the tokenizer for better text generation.
* Generate longer and more meaningful responses.
* Improve the transformer model for better performance.
* Allow users to fine-tune the model on their own datasets.
* Add Docker support for easier setup and deployment.

---

## Project Status

Completed successfully as part of a Machine Learning Internship project and later enhanced with a FastAPI backend, OpenAI-compatible APIs and OpenWebUI integration.
