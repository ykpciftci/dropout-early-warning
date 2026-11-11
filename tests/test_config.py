import pytest
from src import config


def test_early_models_have_no_second_semester_columns():
    """
    Tests if a 2nd semester feature appears in enrollment or first semester features
    """
    for feat in config.SEMESTER_1_MODEL_FEATURES:
        assert "2nd sem" not in feat
    for feat in config.ENROLLMENT_MODEL_FEATURES:
        assert "2nd sem" not in feat


def test_enrollment_features():
    """
    Tests if a semester feature appears in enrollment features
    """
    for feat in config.ENROLLMENT_MODEL_FEATURES:
        assert " sem " not in feat


def test_debt_tuition_scholar():
    """
    Tests if debtor, tuition or scholarship features appear in enrollment or 1st semester features.
    """
    checklist = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
    for feat in checklist:
        assert feat not in config.ENROLLMENT_MODEL_FEATURES
        assert feat not in config.SEMESTER_1_MODEL_FEATURES


def test_each_feature_is_either_categorical_or_numerical():
    """
    Test if all features in categorical features list or numerical features list and not in both.
    """
    all_features = (
        config.ENROLLMENT_FEATURES
        + config.SEMESTER_1_FEATURES
        + config.SEMESTER_2_FEATURES
    )
    for feat in all_features:
        assert (feat in config.CATEGORICAL_FEATURES) != (
            feat in config.NUMERICAL_FEATURES
        )


def test_no_duplicates():
    """
    “Model feature lists contain no duplicates.”
    """
    assert len(config.ENROLLMENT_MODEL_FEATURES) == len(
        set(config.ENROLLMENT_MODEL_FEATURES)
    )
    assert len(config.SEMESTER_1_MODEL_FEATURES) == len(
        set(config.SEMESTER_1_MODEL_FEATURES)
    )
