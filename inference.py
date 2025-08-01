# inference.py
from model.embedder import get_embeddings

def predict(query, classify_model, embed_model, tokenizer, device, k=1):
    emb = get_embeddings([query], embed_model, tokenizer, device, prefix="query: ")
    y_prob = classify_model.predict_propba(emb)
    y_pred = (y_prob > 0).astype(int)
    return y_pred
