"""Two fixed model definitions; no parameter search."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic():
    """Learn imputation and scaling from the training data only."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000),
    )


def make_tree():
    """Use September explicitly for early stopping, with AP as the score."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        scoring="average_precision",
        n_iter_no_change=20,
        random_state=0,
    )
