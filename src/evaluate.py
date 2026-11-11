import joblib
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import load_data, split_data


def evaluate(df, name, features, model_path):
    _, X_test, _, y_test = split_data(
        df=df,
        features=features,
        target=config.TARGET,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
    )
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(f"{name} - missed dropouts: {matrix[1, 0]}")

    print(f"Classification for {name}:")
    print(classification_report(y_test, y_pred))


def main():
    df = load_data(path=config.DATA_PATH, target=config.TARGET)
    evaluate(
        df,
        "Enrollment model",
        config.ENROLLMENT_MODEL_FEATURES,
        config.ENROLLMENT_MODEL_PATH,
    )
    evaluate(
        df,
        "Semester 1 model",
        config.SEMESTER_1_MODEL_FEATURES,
        config.SEMESTER_1_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
