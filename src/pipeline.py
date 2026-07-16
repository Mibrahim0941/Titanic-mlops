from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OneHotEncoder

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RF_PARAMS,
    AGE_BINS,
)


def create_pipeline():
    pipe = Pipeline([
        (
            "num_imputer",
            MeanMedianImputer(
                imputation_method="median",
                variables=NUMERICAL_FEATURES
            ),
        ),

        (
            "cat_imputer",
            CategoricalImputer(
                imputation_method="frequent",
                variables=["embarked"],
            ),
        ),

        (
            "age_bins",
            EqualFrequencyDiscretiser(
                q=AGE_BINS,
                variables=["age"],
            ),
        ),

        (
            "one_hot",
            OneHotEncoder(
                variables=CATEGORICAL_FEATURES,
                ignore_format=True,
            ),
        ),

        (
            "model",
            RandomForestClassifier(**RF_PARAMS),
        ),
    ])

    return pipe