from models.PoSGRU import PoSGRU
import pickle
import torch

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_model(checkpoint_path, vocab_size, label_size, embed_dim, hidden_dim, num_layers, residual, embed_init=None):
    model = PoSGRU(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=label_size,
        residual=residual,
        embed_init=embed_init
    )
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def tokenize(sentence):
    return sentence.strip().split()

def numeralize(tokens, vocab):
    if "<unk>" in vocab.word2idx:
        unk_idx = vocab.word2idx["<unk>"]
    else:
        unk_idx = 0
        
    return [vocab.word2idx.get(token.lower(), unk_idx) for token in tokens]

def denumeralize(indices, vocab):
    return [vocab.idx2label.get(idx, "UNK") for idx in indices]

def predict(sentence, model, vocab):
    tokens = tokenize(sentence)
    x = numeralize(tokens, vocab)
    x_tensor = torch.tensor(x, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_tensor)
        predictions = torch.argmax(logits, dim=2)
        pred_labels = predictions.squeeze(0).tolist()

    tags = denumeralize(pred_labels, vocab)
    return list(zip(tokens, tags))

def main():
    model_path = "model.pt"
    vocab_path = "vocab.pkl"
    vocab = load_vocab(vocab_path)
    checkpoint = torch.load(model_path, map_location="cuda")
    config = checkpoint['config']

    model = load_model(
        checkpoint_path=model_path,
        vocab_size=vocab.lenWords(),
        label_size=vocab.lenLabels(),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["layers"],
        residual=config["residual"]
    )

    print("enter a sentence ")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == "exit":
            break
        result = predict(sentence, model, vocab)
        for word, tag in result:
            print(f"{word}\t{tag}")
        print()

if __name__ == "__main__":
    main()
