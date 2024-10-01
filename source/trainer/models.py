from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocess import SavgolWrapper

savgol_xgb = Pipeline(
    [
        ("savgol", SavgolWrapper()),
        ("xgb", XGBClassifier(random_state=0)),
    ]
)

MODELS = {
    "savgol_xgb": savgol_xgb,
}
