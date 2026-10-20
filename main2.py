
from transformers import pipeline, AutoTokenizer

print("Loading model...")

model_name = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# 1. Load model and tokenizer
classifier = pipeline(
    "sentiment-analysis",
    model=model_name
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded!")

# 2. Input reviews
reviews = [
    "This product is amazing",
    "I hate this product",
    "The quality is okay",
    "Worth it Product"

]

# 3. Understand tokenization
for review in reviews:
    print("\nReview:", review)

    tokens = tokenizer.tokenize(review)
    print("Tokens:", tokens)

    encoded = tokenizer(
        review,
        return_tensors="pt"
    )

    print("Input IDs:", encoded["input_ids"])
    print("Attention Mask:", encoded["attention_mask"])

# 4. Predict sentiments
print("\nPredicting sentiments...")

results = classifier(reviews)

for review, result in zip(reviews, results):
    print("\nReview:", review)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))
    print("-" * 40)
# Embedding
print("Checking embeddings...")

inputs = tokenizer(
    "I hate this product",
    return_tensors="pt"
)

print("Input IDs:", inputs["input_ids"])

embeddings = classifier.model.distilbert.embeddings(
    input_ids=inputs["input_ids"]
)

print("Embedding shape:", embeddings.shape)



import torch

from transformers import AutoModelForSequenceClassification

model_name = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="eager"
)

model.eval()

model.eval()

text = "I hate this product"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )

print("Tokens:", tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
))

print("Number of attention layers:", len(outputs.attentions))

print("Attention shape:", outputs.attentions[0].shape)

# First layer, first attention head
attention = outputs.attentions[0][0, 0]

print("\nAttention weights:")
print(attention)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

for token, row in zip(tokens, attention):
    print(f"\nToken: {token}")

    for other_token, weight in zip(tokens, row):
        print(f"  -> {other_token}: {weight.item():.3f}")

# k,Q,V 
# 1. Imports
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 2. Model load karna
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    attn_implementation="eager"
)
model.eval()

# 3. Tumhara existing input code
text = "I hate this product"
inputs = tokenizer(text, return_tensors="pt")

# 4. YAHAN Q, K, V WALA CODE ADD KARO
layer = model.distilbert.transformer.layer[0]

with torch.no_grad():
    hidden_states = model.distilbert.embeddings(
        input_ids=inputs["input_ids"]
    )

    attention_layer = layer.attention
    q = attention_layer.q_lin(hidden_states)
    k = attention_layer.k_lin(hidden_states)
    v = attention_layer.v_lin(hidden_states)

print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)



import math

# DistilBERT has 12 attention heads
num_heads = 12
head_dim = 768 // num_heads

# Split Q, K, V into 12 heads
q_heads = q.view(1, 6, num_heads, head_dim).transpose(1, 2)
k_heads = k.view(1, 6, num_heads, head_dim).transpose(1, 2)
v_heads = v.view(1, 6, num_heads, head_dim).transpose(1, 2)

print("Q heads shape:", q_heads.shape)
print("K heads shape:", k_heads.shape)
print("V heads shape:", v_heads.shape)

# Step 1: Q × K transpose
scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))

# Step 2: Scaling
scores = scores / math.sqrt(head_dim)

# Step 3: Softmax
attention_weights = torch.softmax(scores, dim=-1)

print("Attention weights shape:", attention_weights.shape)
print("First head attention weights:")
print(attention_weights[0, 0])

# 5. Iske baad tumhara existing attention heatmap code

import matplotlib.pyplot as plt
import seaborn as sns

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

sns.heatmap(
    attention.numpy(),
    xticklabels=tokens,
    yticklabels=tokens,
    annot=True,
    fmt=".2f"
)

plt.xlabel("Key tokens")
plt.ylabel("Query tokens")
plt.title("DistilBERT Self-Attention")
plt.tight_layout()
plt.show()



# Step 4: Attention weights × V
attention_output = torch.matmul(attention_weights, v_heads)

print("Attention output shape:", attention_output.shape)
print("\nFirst head output:")
print(attention_output[0, 0])

# Combine all 12 heads
combined_output = attention_output.transpose(1, 2).contiguous().view(1, 6, 768)

print("\nCombined output shape:", combined_output.shape)
