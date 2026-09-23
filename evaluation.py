import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR 

def evaluate(run_id):
    test_data = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/HeartAttack-model")

    predictions = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("roc_auc", roc_auc)

    print(f"Evaluation completed | ROC score = {roc_auc:.3f}")

    return roc_auc

if __name__ == "__main__":
    evaluate("e5d2599df7e5479d9279357377b03952")