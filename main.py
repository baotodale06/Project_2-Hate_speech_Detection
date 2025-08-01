# main.py
from train import main as train_model
from inference import predict

def format_result(y_pred):
    labels = ["L", "G", "B", "T", "Other", "Not Related"]
    return " ".join([f"{l}:{y}" for l, y in zip(labels, y_pred[0])])

if __name__ == "__main__":
    model, embed_model, tokenizer, device = train_model()
    while True:
        query = input("Enter your query: ")
        y_pred = predict(query, model, embed_model, tokenizer, device)
        print("Prediction:", format_result(y_pred))

