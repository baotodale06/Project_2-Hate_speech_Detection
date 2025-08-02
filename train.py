# train.py
from model.embedder import init_embed_model, get_embeddings
from model.knn_classifier import train_knn_classifier
from utils.data_loader import load_dataset
from sklearn.metrics import classification_report, jaccard_score, f1_score, hamming_loss

def main():
    embed_model, tokenizer, device = init_embed_model()
    # texts_train, labels_train, text_test, label_test = load_dataset(path="data/track2_train/track2_simple.csv")
    texts_train, labels_train, text_test, label_test = load_dataset(path="data/track2_train/track2_augmented_enhanced.csv")
    X_embed = get_embeddings(texts_train, embed_model, tokenizer, device)
    model = train_knn_classifier(X_embed, labels_train)
    X_test_embed = get_embeddings(text_test, embed_model, tokenizer, device)
    # Evaluate the model here if needed
    X_test_pred = model.predict(X_test_embed)
    print(classification_report(label_test, X_test_pred))
    # jaccard = jaccard_score(label_test, X_test_pred, average='macro')
    # f1 = f1_score(label_test, X_test_pred, average='macro')
    # hamming = hamming_loss(label_test, X_test_pred)
    # print(f"Jaccard Score: {jaccard}")
    # print(f"F1 Score: {f1}")
    # print(f"Hamming Loss: {hamming}")
    # Save model and embeddings here if needed
    return model, embed_model, tokenizer, device

if __name__ == "__main__":
    main()
