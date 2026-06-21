import numpy as np
import matplotlib.pyplot as plt

max_len = 50
d_model = 32

pe = np.zeros((max_len, d_model))

for pos in range(max_len):
    for i in range(0, d_model, 2):
        pe[pos, i] = np.sin(
            pos / (10000 ** (i / d_model))
        )

        pe[pos, i + 1] = np.cos(
            pos / (10000 ** (i / d_model))
        )

plt.figure(figsize=(10, 6))
plt.imshow(pe, aspect='auto')
plt.colorbar()

plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")

plt.savefig("positional_encoding.png")

print("Saved positional_encoding.png")
