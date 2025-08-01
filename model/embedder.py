# model/embedder.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def init_embed_model(model_name="intfloat/multilingual-e5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), tokenizer, device

def get_embeddings(texts, model, tokenizer, device, batch_size=32, prefix="passage: "):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]

        batch = [prefix + t for t in batch]

        batch_dict = tokenizer(batch,
                              max_length=512,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')
        
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            output = model(**batch_dict)
            pooled = average_pool(output.last_hidden_state, batch_dict["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)
