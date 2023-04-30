from preprocess import DataPreprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlflow

class LogisticRegressionModel():
    def __init__(self):
        # 데이터 정제 클래스 정의
        preprocess = DataPreprocess()
        preprocess_data = preprocess.data_split()

        # 데이터 정제
        self.x_train, self.x_test = preprocess_data[0], preprocess_data[2]
        self.y_train, self.y_test = preprocess_data[3], preprocess_data[5]

        # 모델 파라미터 값 설정
        self.penalty = "l2"
        self.C = 2.0
        self.random_state = None

        # 모델 정의(Logistic Regression)
        self.model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            random_state=self.random_state
            )


    def main(self):
        # MLflow 실험 이름 정의
        mlflow.set_experiment("LogisticRegression_Experiment")

        # 모델 훈련 및 평가
        train_model = self.model.fit(self.x_train, self.y_train)
        eval_acc = train_model.score(self.x_test, self.y_test)
        preds = train_model.predict(self.x_test)
        auc_score = roc_auc_score(self.y_test, preds)
        print(f"AUC Score: {auc_score:.3%}")
        print(f"Eval Accuracy: {eval_acc:.3%}")

        # MLflow에 파라미터 값 로깅
        mlflow.log_param("penalty", self.penalty)
        mlflow.log_param("C", self.C)
        mlflow.log_param("random_state", self.random_state)

        # MLflow에 평가 지표 로깅
        mlflow.log_metric("eval_acc", eval_acc)
        mlflow.log_metric("auc_score", auc_score)

        # 훈련된 모델을 MLflow 로깅
        mlflow.sklearn.log_model(train_model, "lr_model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


if __name__ == '__main__':
    lr_model = LogisticRegressionModel()
    lr_model.main()




