import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_dummy_model():
    # Dummy dataset
    data = {
        "amount": [9.99, 15.99, 200, 12.99, 1200],
        "frequency": [30, 30, 1, 30, 1],  # 30 = monthly, 1 = one-off
        "label": [1, 1, 0, 1, 0]  # 1 = subscription, 0 = non-subscription
    }
    df = pd.DataFrame(data)

    X = df[["amount", "frequency"]]
    y = df["label"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

if __name__ == "__main__":
    model = train_dummy_model()
    print("Dummy RandomForest model trained successfully!")
