import torch
import struct
import os
from gpt_model_copy import GPTLanguageModel

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt_best_model.pt")

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = GPTLanguageModel(
    checkpoint["vocab_size"],
    checkpoint["n_embd"],
    checkpoint["n_head"],
    checkpoint["n_layer"],
    checkpoint["block_size"],
    checkpoint["dropout"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

stoi = checkpoint["stoi"]
block_size = checkpoint["block_size"]

def get_sentence_embedding(text: str):
    ids = []
    for c in text:
        if c in stoi:
            ids.append(stoi[c])
    ids = ids[:block_size]

    if len(ids) == 0:
        raise ValueError(f"no encodable characters in: {text!r}")

    context = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        num_tokens = context.shape[1]

        tok_emb = model.token_embedding_table(context)

        nn_list = []
        for i in range(0, num_tokens):
            nn_list.append(i)
        position_ids = torch.tensor(nn_list, dtype=torch.long)
        position_emb = model.position_embedding_table(position_ids)

        token_repr = tok_emb + position_emb

        for block in model.blocks:
            token_repr = block(token_repr)

        token_repr = model.final_normalization(token_repr)

        token_repr_list = token_repr[0].tolist()
        n_embd = len(token_repr_list[0])

        embedding = []
        for coord in range(n_embd):
            total = 0
            for token_vector in token_repr_list:
                total += token_vector[coord]
            embedding.append(total / num_tokens)

    return embedding


if __name__ == "__main__":
    INPUT_TEXT_PATH = os.path.join(os.path.dirname(__file__), "input.txt")

    with open(INPUT_TEXT_PATH, "r") as f:
        text = f.read()

    words = []
    current_word = ""

    for char in text:
        if char.isalpha() or char == "'":
            current_word += char
        else:
            if current_word != "":
                words.append(current_word)
            current_word = ""

    if current_word != "":
        words.append(current_word)

    words = [w.lower() for w in words]
    unique_words = sorted(set(words))

    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "embeddings.bin")
    WORDS_PATH = os.path.join(os.path.dirname(__file__), "unique_words.txt")

    with open(WORDS_PATH, "w") as f:
        for word in unique_words:
            f.write(word + "\n")

    with open(OUTPUT_PATH, "wb") as f:
        for i, word in enumerate(unique_words):
            embedding = get_sentence_embedding(word)

            f.write(struct.pack("i", i))                            # id (int32)
            f.write(struct.pack("i", len(embedding)))               # dim (int32)
            f.write(struct.pack(f"{len(embedding)}f", *embedding))  # data (floats)

            if i % 500 == 0:
                print(f"embedded {i}/{len(unique_words)}")

    print(f"done - wrote {len(unique_words)} embeddings to {OUTPUT_PATH}")
