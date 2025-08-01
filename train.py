# train.py
from model.embedder import init_embed_model, get_embeddings
from model.knn_classifier import train_knn_classifier
from utils.data_loader import load_dataset

def main():
    embed_model, tokenizer, device = init_embed_model()
    texts, labels = load_dataset()
    X_embed = get_embeddings(texts, embed_model, tokenizer, device)
    model = train_knn_classifier(X_embed, labels)
    # Save model and embeddings here if needed
    return model, embed_model, tokenizer, device

if __name__ == "__main__":
    main()
