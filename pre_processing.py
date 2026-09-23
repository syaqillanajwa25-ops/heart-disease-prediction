import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess():
    df = pd.read_csv("DataIngested/Heart Attack Data Set.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    ohe_cols = ['cp','restecg','thal']
    skip_cols = [col for col in X.columns if col not in ohe_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
            ("num", "passthrough", skip_cols)])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

    train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

if __name__ == "__main__":
    preprocess()