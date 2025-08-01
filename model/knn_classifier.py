# model/knn_classifier.py
from faissknn import FaissKNNMultilabelClassifier

def train_knn_classifier(X, y, k=3):
    model = FaissKNNMultilabelClassifier(n_neighbors=k, 
                                         n_classes=y.shape[1], 
                                         device="cpu")
    model.fit(X, y)
    return model
