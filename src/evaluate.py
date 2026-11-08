import joblib
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import load_data, split_data


def main():
    df = load_data(path=config.DATA_PATH, target=config.TARGET)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=config.ENROLLMENT_MODEL_FEATURES,
        target=config.TARGET,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
    )
    model = joblib.load(config.ENROLLMENT_MODEL_PATH)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(
        f"Number of students who are predicted to graduate but drop out: {matrix[1,0]}"
    )
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
