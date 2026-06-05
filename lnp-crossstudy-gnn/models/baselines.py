"""
Descriptor-based baseline models for LNP property prediction.

All baselines use the curated Tier-1 feature set (RDKit + pruned Mordred + ratios
+ process parameters, ~475 features) and are evaluated under 39-fold LOSO-CV
with Z-normalised targets — the same protocol as the GNN.

Hyperparameters were tuned with Optuna using inner LOSO-CV on training studies.
Best parameters are stored in results/best_hyperparameters.json.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb


def make_rf_default():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )


def make_rf_tuned():
    """Optuna-tuned RF — best inner-LOSO hyperparameters from Phase 2A."""
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=18,
        min_samples_split=10,
        max_features="log2",
        random_state=42,
        n_jobs=-1,
    )


def make_xgboost_tuned():
    """Optuna-tuned XGBoost — extreme regularisation reflects near-zero signal."""
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=8.75,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def make_lightgbm_tuned():
    """Optuna-tuned LightGBM — shallow trees to prevent inter-study overfitting."""
    return lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_svr():
    """RBF-kernel SVR — requires StandardScaler on features before use."""
    return SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")


# Registry for easy iteration
BASELINE_MODELS = {
    "RF (default)": make_rf_default,
    "RF (tuned)": make_rf_tuned,
    "XGBoost (tuned)": make_xgboost_tuned,
    "LightGBM (tuned)": make_lightgbm_tuned,
    "SVR (RBF)": make_svr,
}
