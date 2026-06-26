import json
import torch
import torch.nn as nn
import torch.nn.functional as F

with open("word_to_idx.json") as f:
    word_to_idx = json.load(f)

with open("idx_to_word.json") as f:
    idx_to_word = {
        int(k): v
        for k, v in json.load(f).items()
    }


class MicroGPT(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim=64,
                 num_heads=2,
                 num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


model = MicroGPT(len(word_to_idx))

model.load_state_dict(
    torch.load(
        "models/micro_gpt_model.pth",
        map_location="cpu"
    )
)

model.eval()


def generate(prompt, max_new_tokens=15, k=5):

    generated_words = prompt.lower().split()

    tokens = [
        word_to_idx[word]
        for word in generated_words
        if word in word_to_idx
    ]

    for _ in range(max_new_tokens):

        input_tokens = tokens[-5:]

        while len(input_tokens) < 5:
            input_tokens.insert(0, 0)

        x = torch.tensor([input_tokens])

        with torch.no_grad():
            logits = model(x)

        probabilities = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k)

        prediction = top_indices[
            0,
            torch.multinomial(top_probs[0], 1)
        ].item()

        predicted_word = idx_to_word[prediction]

        generated_words.append(predicted_word)
        tokens.append(prediction)

    return " ".join(generated_words)


if __name__ == "__main__":
    prompt = input("Enter Prompt: ")
    print(generate(prompt))
