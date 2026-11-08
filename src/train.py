import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

from src import config
from src.data import load_data, split_data


def build_pipeline(categorical, numerical):
    # balanced: missing a future dropout costs more than a false alarm.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced")),
        ]
    )

    return pipeline


def cross_validate_model(name, model, random_state, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
    scores = cross_validate(model, X, y, cv=cv, scoring=["recall", "f1"])
    recall_scores = scores["test_recall"]
    f1_scores = scores["test_f1"]
    print(f"{name} recall: {recall_scores.mean():.3f} +/- {recall_scores.std():.3f}")
    print(f"{name} f1_score: {f1_scores.mean():.3f} +/- {f1_scores.std():.3f}")


def main():
    df = load_data(path=config.DATA_PATH, target=config.TARGET)
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        features=config.ENROLLMENT_MODEL_FEATURES,
        target=config.TARGET,
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
    )
    numeric_features = [f for f in config.NUMERICAL_FEATURES if f in X_train.columns]
    categoric_features = [
        f for f in config.CATEGORICAL_FEATURES if f in X_train.columns
    ]

    model = build_pipeline(numerical=numeric_features, categorical=categoric_features)

    cross_validate_model(
        "Enrollment Model",
        model=model,
        random_state=config.RANDOM_STATE,
        X=X_train,
        y=y_train,
    )

    model.fit(X_train, y_train)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.ENROLLMENT_MODEL_PATH)
    print(f"Model saved to {config.ENROLLMENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
