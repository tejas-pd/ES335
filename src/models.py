"""Model definitions shared by the evaluation and website-export scripts."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CAT = ["brand", "brand_model", "fuel", "transmission"]
NUM = ["age", "km"]
SPEC = ["engine", "power", "mileage", "seats", "owner"]  # only used in the ablation
SEED = 42


def _num_features(X):
    X = np.asarray(X, dtype=float)
    age, km = X[:, 0], X[:, 1]
    return np.column_stack([age, age ** 2, np.log1p(km)])


def _onehot():
    # Rare models (<5 training listings) share one "infrequent" column; models never
    # seen in training map to it as well instead of raising an error.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                         sparse_output=True)


class LogTarget(BaseEstimator, RegressorMixin):
    """Fit on log(price), predict exp(.) -> a median-type prediction in rupees."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.model.predict(X))


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Proposal baseline: predict the average training price for every car."""

    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class LookupBaseline(BaseEstimator, RegressorMixin):
    """Domain heuristic: median training price of the same brand+model,
    falling back to the brand median, then the overall median. Ignores age/km."""

    def fit(self, X, y):
        d = X.assign(_y=np.asarray(y))
        self.model_ = d.groupby("brand_model")._y.median().to_dict()
        self.brand_ = d.groupby("brand")._y.median().to_dict()
        self.global_ = float(d._y.median())
        return self

    def predict(self, X):
        out = X.brand_model.map(self.model_)
        out = out.fillna(X.brand.map(self.brand_)).fillna(self.global_)
        return out.to_numpy(dtype=float)


def make_linear(alpha=1.0):
    pre = ColumnTransformer([
        ("cat", _onehot(), CAT),
        ("num", make_pipeline(FunctionTransformer(_num_features), StandardScaler()), NUM),
    ])
    return LogTarget(Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))]))


def make_forest(min_samples_leaf=2, max_features=0.5, n_estimators=120, max_depth=None,
                with_specs=False):
    parts = [("cat", _onehot(), CAT), ("num", "passthrough", NUM)]
    if with_specs:
        # Medians are learned from the training fold only (inside the pipeline).
        parts.append(("spec", SimpleImputer(strategy="median", add_indicator=True), SPEC))
    pre = ColumnTransformer(parts)
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                               max_features=max_features, max_depth=max_depth,
                               n_jobs=-1, random_state=SEED)
    return LogTarget(Pipeline([("pre", pre), ("reg", rf)]))


LINEAR_GRID = [{"alpha": a} for a in (0.1, 1.0, 10.0)]
FOREST_GRID = [{"min_samples_leaf": l, "max_features": f} for l in (2, 5) for f in (0.3, 0.6)]
