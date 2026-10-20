
from transformers import pipeline

print("Loading model...")

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline(
    "sentiment-analysis",
    model=model_name
)

print("Model loaded!")


import pandas as pd

# CSV file read karo
df = pd.read_csv("reviews.csv")


print("First 5 reviews:")
print(df.head())

# Reviews par sentiment predict karo
results = classifier(
    df["review"].astype(str).tolist()
)

df["sentiment"] = [r["label"] for r in results]
df["confidence"] = [round(r["score"], 4) for r in results]

# Results save karo
df.to_csv("sentiment_results.csv", index=False)



print("\nSentiment analysis complete!")
print(df.head())
print("\nResults saved to sentiment_results.csv")

import matplotlib.pyplot as plt

# Count each sentiment
sentiment_counts = df["sentiment"].value_counts()

print("\nSentiment Summary:")
print(sentiment_counts)

# Plot sentiment distribution
sentiment_counts.plot(kind="bar")

plt.title("Product Review Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



result = classifier("I hate this product")

print("Final prediction:", result)

# Model ke classification logits
inputs = tokenizer("I hate this product", return_tensors="pt")

import torch

with torch.no_grad():
    outputs = classifier.model(**inputs)

print("Logits:", outputs.logits)
print("Probabilities:", torch.softmax(outputs.logits, dim=-1))