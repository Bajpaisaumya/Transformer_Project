# Transformer Internals & Sentiment Analysis

## About the Project

This project explores how Transformer models work internally and applies a pretrained DistilBERT model to product review sentiment analysis.

It contains hands-on experiments with tokenization, embeddings, Query-Key-Value (QKV) projections, self-attention, and a practical CSV-based sentiment classification pipeline.

## Project Components

### 1. Transformer Internals (`main2.py`)

- Tokenization and input IDs
- Token embeddings and tensor shapes
- Query, Key, and Value projections
- Multi-head attention concepts
- Scaled dot-product attention
- Attention weights and attention output

### 2. Sentiment Analysis (`main.py`)

- Reads product reviews from `reviews.csv`
- Predicts positive or negative sentiment using DistilBERT
- Adds sentiment labels and confidence scores
- Saves predictions to `sentiment_results.csv`
- Visualizes sentiment counts using Matplotlib

## Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- Pandas
- Matplotlib

## Pretrained Model

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

This model performs binary sentiment classification: POSITIVE or NEGATIVE.

## Project Structure

```text
Transformer_Project/
├── main.py
├── main2.py
├── reviews.csv
├── sentiment_results.csv
├── requirement.txt
└── README.md
```

## Setup

Install the required libraries:

```bash
pip install -r requirement.txt
```

## Run the Project

For CSV-based sentiment analysis:

```bash
python main.py
```

For Transformer internals experiments:

```bash
python main2.py
```

The pretrained model may need to download from Hugging Face the first time it runs.

## Output

The sentiment analysis pipeline produces a CSV containing each review, its predicted sentiment, and its confidence score. It also displays a bar chart showing the sentiment distribution.

**Note:** Confidence scores do not guarantee that a prediction is correct. The model is designed for binary sentiment classification and does not provide a neutral label.

## Future Improvements

- Evaluate predictions using a labeled dataset
- Test on a larger collection of product reviews
- Compare multiple pretrained sentiment models
- Explore embeddings and attention visualizations
