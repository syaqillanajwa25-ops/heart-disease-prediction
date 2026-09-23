import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train():
    mlflow.set_experiment("HeartAttack-MLflow-Pipeline")

    train_data = pd.read_csv("train.csv")
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    with mlflow.start_run() as run:
        model = RandomForestClassifier(criterion="entropy", max_depth= 5, 
                                       max_features= 'sqrt', min_samples_leaf= 1, 
                                       min_samples_split= 2, n_estimators= 100)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model=model, 
                                 artifact_path="HeartAttack-model", 
                                 registered_model_name="HeartAttackModel") 
        return run.info.run_id  
        
if __name__ == "__main__":
    train()

