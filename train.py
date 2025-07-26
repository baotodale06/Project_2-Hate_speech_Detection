from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from faissknn import FaissKNNMultilabelClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# x, y = make_classification()
df = pd.read_csv("data/track2_train/track2_augmented_enhanced.csv")
# df = pd.read_csv("data/track2_train/track2_augmented_simple.csv")
X = df['content'].values
y = df.iloc[:,2:].values
X, y = make_multilabel_classification()
x_train, x_test, y_train, y_test = train_test_split(X, y)
model = FaissKNNMultilabelClassifier(
    n_neighbors=5,
    n_classes=None,
    device="cpu"
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test) # (N,)
y_proba = model.predict_proba(x_test) # (N, C)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))