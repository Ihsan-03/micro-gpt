import pandas as pd
import re
import json

df = pd.read_csv(
    "chatbot_conversations.csv",
    nrows=50000
)

messages = df["message"].astype(str)

text = " ".join(messages)

text = text.lower()

text = re.sub(r"[^a-z0-9\s]", "", text)

tokens = text.split()

vocab = sorted(set(tokens))

word_to_idx = {
    word: idx
    for idx, word in enumerate(vocab)
}

idx_to_word = {
    str(idx): word
    for word, idx in word_to_idx.items()
}

encoded = [
    word_to_idx[token]
    for token in tokens
]

with open("word_to_idx.json", "w") as f:
    json.dump(word_to_idx, f, indent=4)

with open("idx_to_word.json", "w") as f:
    json.dump(idx_to_word, f, indent=4)

with open("encoded_tokens.txt", "w") as f:
    f.write(" ".join(map(str, encoded)))

print("Saved vocabulary and encoded dataset.")
