"""
Centralized Machine Learning Algorithm Registry for Zemo.

Provides:
- 7 Classification Algorithms (Logistic Regression, Decision Tree, Random Forest,
  KNN, SVC, Gaussian Naive Bayes, Gradient Boosting)
- 8 Regression Algorithms (Linear Regression, Ridge, Lasso, Decision Tree,
  Random Forest, KNN, SVR, Gradient Boosting)
- 3 Clustering Algorithms (K-Means, DBSCAN, Agglomerative Clustering)
- Strict hyperparameter validation, typing, bounded ranges, and enum constraints.
- Secure instantiation without dynamic execution or arbitrary Python code.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from core.exceptions import AppValidationError as ValidationError
except ImportError:
    from backend.core.exceptions import AppValidationError as ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter and Algorithm Specifications
# ---------------------------------------------------------------------------


class HyperparameterSpec(BaseModel):
    """Specification and validation constraints for an algorithm hyperparameter."""

    name: str
    param_type: Literal["int", "float", "str", "bool", "none", "choice"]
    default: Any
    description: str
    options: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allow_none: bool = False


class AlgorithmInfo(BaseModel):
    """Full metadata and configuration schema for a machine learning algorithm."""

    id: str
    display_name: str
    task: Literal["classification", "regression", "clustering"]
    is_supervised: bool
    description: str
    family: str
    tags: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, HyperparameterSpec]
    default_hyperparameters: Dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Centralized Algorithm Registry Engine
# ---------------------------------------------------------------------------


class AlgorithmRegistry:
    """
    Centralized registry of supported machine learning algorithms.
    Guarantees that user configurations are strictly validated and only
    safe, registered Scikit-learn estimator classes are instantiated.
    """

    def __init__(self):
        self._algorithms: Dict[str, Tuple[AlgorithmInfo, Type[BaseEstimator]]] = {}
        self._aliases: Dict[str, Dict[str, str]] = {}
        self._register_default_algorithms()

    def register(
        self,
        info: AlgorithmInfo,
        estimator_cls: Type[BaseEstimator],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a new algorithm in the registry."""
        self._algorithms[info.id] = (info, estimator_cls)

        # Register aliases mapped by task
        if aliases:
            for alias in aliases:
                key = alias.lower().replace("_", "-")
                if key not in self._aliases:
                    self._aliases[key] = {}
                self._aliases[key][info.task] = info.id

    def resolve_algorithm_id(self, identifier: str, task: Optional[str] = None) -> str:
        """
        Resolves an algorithm ID or alias (e.g. 'random-forest' for task='classification'
        resolves to 'random_forest_classifier').
        """
        raw_id = identifier.strip().lower()
        dash_key = raw_id.replace("_", "-")

        # Prioritize task-specific alias if task is specified
        if task and dash_key in self._aliases and task in self._aliases[dash_key]:
            return self._aliases[dash_key][task]

        if raw_id in self._algorithms:
            return raw_id

        # Normalize with underscores
        underscore_id = raw_id.replace("-", "_")
        if underscore_id in self._algorithms:
            return underscore_id

        # Check aliases
        if dash_key in self._aliases:
            task_map = self._aliases[dash_key]
            if task and task in task_map:
                return task_map[task]
            # If no task specified, return first match
            return next(iter(task_map.values()))


        raise ValidationError(
            f"Unsupported algorithm '{identifier}'. Available algorithms: {list(self._algorithms.keys())}"
        )

    def get(self, identifier: str, task: Optional[str] = None) -> AlgorithmInfo:
        """Retrieve algorithm specification metadata."""
        algo_id = self.resolve_algorithm_id(identifier, task)
        return self._algorithms[algo_id][0]

    def list_algorithms(self, task: Optional[str] = None) -> List[AlgorithmInfo]:
        """List all algorithms, optionally filtered by task."""
        algos = [info for info, _ in self._algorithms.values()]
        if task:
            task_norm = task.lower().strip()
            algos = [a for a in algos if a.task == task_norm]
        return algos

    def validate_hyperparameters(
        self,
        identifier: str,
        hyperparameters: Dict[str, Any],
        task: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validates user-supplied hyperparameters against the algorithm specification.
        Returns (is_valid, sanitized_params, errors).
        """
        algo_id = self.resolve_algorithm_id(identifier, task)
        info, _ = self._algorithms[algo_id]
        errors: List[str] = []
        sanitized: Dict[str, Any] = dict(info.default_hyperparameters)

        for key, val in hyperparameters.items():
            if key not in info.hyperparameters:
                errors.append(f"Unknown parameter '{key}' for algorithm '{info.display_name}'.")
                continue

            spec = info.hyperparameters[key]

            # Allow None if permitted
            if val is None:
                if spec.allow_none:
                    sanitized[key] = None
                    continue
                else:
                    errors.append(f"Parameter '{key}' cannot be null.")
                    continue

            # Type checking and coercion
            try:
                if spec.param_type == "int":
                    val_int = int(val)
                    if spec.min_value is not None and val_int < spec.min_value:
                        errors.append(f"Parameter '{key}' must be >= {spec.min_value} (received {val_int}).")
                    if spec.max_value is not None and val_int > spec.max_value:
                        errors.append(f"Parameter '{key}' must be <= {spec.max_value} (received {val_int}).")
                    sanitized[key] = val_int

                elif spec.param_type == "float":
                    val_float = float(val)
                    if spec.min_value is not None and val_float < spec.min_value:
                        errors.append(f"Parameter '{key}' must be >= {spec.min_value} (received {val_float}).")
                    if spec.max_value is not None and val_float > spec.max_value:
                        errors.append(f"Parameter '{key}' must be <= {spec.max_value} (received {val_float}).")
                    sanitized[key] = val_float

                elif spec.param_type == "bool":
                    if isinstance(val, bool):
                        sanitized[key] = val
                    elif isinstance(val, str):
                        sanitized[key] = val.lower() in ("true", "1", "yes")
                    else:
                        sanitized[key] = bool(val)

                elif spec.param_type in ("str", "choice"):
                    val_str = str(val).strip()
                    if spec.options and val_str not in spec.options:
                        errors.append(
                            f"Invalid option '{val_str}' for parameter '{key}'. Allowed options: {spec.options}"
                        )
                    else:
                        sanitized[key] = val_str

            except (ValueError, TypeError) as e:
                errors.append(f"Invalid value '{val}' for parameter '{key}' ({spec.param_type}): {str(e)}")

        # Special conditional algorithm checks
        if algo_id == "logistic_regression":
            solver = sanitized.get("solver")
            penalty = sanitized.get("penalty")
            if solver == "lbfgs" and penalty == "l1":
                errors.append("Solver 'lbfgs' does not support L1 penalty. Use 'saga' or 'liblinear'.")
            if solver == "liblinear" and penalty == "none":
                errors.append("Solver 'liblinear' does not support penalty='none'.")

        is_valid = len(errors) == 0
        return is_valid, sanitized, errors

    def validate_compatibility(
        self,
        identifier: str,
        task: str,
        target_type: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validates whether the selected algorithm is compatible with the given task and target.
        """
        errors: List[str] = []
        try:
            algo_id = self.resolve_algorithm_id(identifier, task)
            info, _ = self._algorithms[algo_id]
        except ValidationError as e:
            return False, [str(e)]

        task_norm = task.lower().strip()
        if info.task != task_norm:
            errors.append(
                f"Algorithm '{info.display_name}' is designed for {info.task.upper()}, "
                f"but task is configured as {task_norm.upper()}."
            )

        if task_norm == "clustering":
            if target_type:
                # Warning / error depending on strictness
                pass
        elif task_norm == "classification":
            if target_type == "numeric" and not info.task == "classification":
                errors.append(f"Target column type '{target_type}' is continuous; use a regression algorithm.")
        elif task_norm == "regression":
            if target_type in ("categorical", "boolean", "text"):
                errors.append(f"Target column type '{target_type}' is discrete/categorical; use a classification algorithm.")

        return len(errors) == 0, errors

    def create_estimator(
        self,
        identifier: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> BaseEstimator:
        """
        Safely instantiates a Scikit-learn estimator instance.
        Validates parameters first to prevent invalid estimator initialization.
        """
        algo_id = self.resolve_algorithm_id(identifier, task)
        info, estimator_cls = self._algorithms[algo_id]

        user_params = hyperparameters or {}
        is_valid, sanitized_params, errors = self.validate_hyperparameters(algo_id, user_params, task)
        if not is_valid:
            raise ValidationError(
                f"Hyperparameter validation failed for '{info.display_name}': {'; '.join(errors)}"
            )

        try:
            return estimator_cls(**sanitized_params)
        except Exception as e:
            raise ValidationError(
                f"Failed to instantiate '{info.display_name}' with parameters {sanitized_params}: {str(e)}"
            )

    # -----------------------------------------------------------------------
    # Registration of all 18 Scikit-learn Algorithms
    # -----------------------------------------------------------------------

    def _register_default_algorithms(self) -> None:
        # ===================================================================
        # 1. CLASSIFICATION ALGORITHMS (7)
        # ===================================================================

        # 1.1 Logistic Regression
        self.register(
            AlgorithmInfo(
                id="logistic_regression",
                display_name="Logistic Regression",
                task="classification",
                is_supervised=True,
                description="Linear model for binary and multiclass classification using a logistic/softmax link function.",
                family="linear",
                tags=["Linear", "Probabilistic", "Fast", "Interpretable"],
                hyperparameters={
                    "C": HyperparameterSpec(name="C", param_type="float", default=1.0, min_value=0.0001, max_value=10000.0, description="Inverse of regularization strength (smaller values specify stronger regularization)"),
                    "solver": HyperparameterSpec(name="solver", param_type="choice", default="lbfgs", options=["lbfgs", "liblinear", "saga", "sag", "newton-cg"], description="Optimization algorithm"),
                    "max_iter": HyperparameterSpec(name="max_iter", param_type="int", default=1000, min_value=10, max_value=10000, description="Maximum number of iterations taken for the solvers to converge"),
                    "penalty": HyperparameterSpec(name="penalty", param_type="choice", default="l2", options=["l2", "l1", "elasticnet", None], allow_none=True, description="Regularization penalty term"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed for reproducibility"),
                },
                default_hyperparameters={"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "penalty": "l2", "random_state": 42},
            ),
            LogisticRegression,
            aliases=["logistic-regression", "logreg"],
        )

        # 1.2 Decision Tree Classifier
        self.register(
            AlgorithmInfo(
                id="decision_tree_classifier",
                display_name="Decision Tree Classifier",
                task="classification",
                is_supervised=True,
                description="Non-parametric supervised hierarchical partition tree with high visual interpretability.",
                family="tree",
                tags=["Tree", "Interpretable", "Non-linear"],
                hyperparameters={
                    "criterion": HyperparameterSpec(name="criterion", param_type="choice", default="gini", options=["gini", "entropy", "log_loss"], description="Function to measure split quality"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=None, min_value=1, max_value=100, allow_none=True, description="Maximum depth of the tree (None for unlimited)"),
                    "min_samples_split": HyperparameterSpec(name="min_samples_split", param_type="int", default=2, min_value=2, max_value=50, description="Minimum number of samples required to split an internal node"),
                    "min_samples_leaf": HyperparameterSpec(name="min_samples_leaf", param_type="int", default=1, min_value=1, max_value=50, description="Minimum number of samples required to be at a leaf node"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"criterion": "gini", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
            ),
            DecisionTreeClassifier,
            aliases=["decision-tree-classifier", "decision-tree"],
        )

        # 1.3 Random Forest Classifier
        self.register(
            AlgorithmInfo(
                id="random_forest_classifier",
                display_name="Random Forest Classifier",
                task="classification",
                is_supervised=True,
                description="Bagged ensemble of decorrelated decision trees delivering competitive generalization and robustness against overfitting.",
                family="ensemble",
                tags=["Ensemble", "Bagging", "Robust", "Competitive"],
                hyperparameters={
                    "n_estimators": HyperparameterSpec(name="n_estimators", param_type="int", default=100, min_value=10, max_value=1000, description="Number of trees in the forest"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=None, min_value=1, max_value=100, allow_none=True, description="Maximum depth of the individual trees"),
                    "min_samples_split": HyperparameterSpec(name="min_samples_split", param_type="int", default=2, min_value=2, max_value=50, description="Minimum samples to split a node"),
                    "min_samples_leaf": HyperparameterSpec(name="min_samples_leaf", param_type="int", default=1, min_value=1, max_value=50, description="Minimum samples required at each leaf"),
                    "max_features": HyperparameterSpec(name="max_features", param_type="choice", default="sqrt", options=["sqrt", "log2", None], allow_none=True, description="Number of features considered per split"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42},
            ),
            RandomForestClassifier,
            aliases=["random-forest-classifier", "random-forest"],
        )

        # 1.4 K-Nearest Neighbors Classifier
        self.register(
            AlgorithmInfo(
                id="knn_classifier",
                display_name="K-Nearest Neighbors Classifier",
                task="classification",
                is_supervised=True,
                description="Instance-based non-parametric classifier voting by metric proximity of nearest neighbors.",
                family="neighbors",
                tags=["Instance-based", "Non-parametric", "Metric"],
                hyperparameters={
                    "n_neighbors": HyperparameterSpec(name="n_neighbors", param_type="int", default=5, min_value=1, max_value=100, description="Number of neighboring points to query"),
                    "weights": HyperparameterSpec(name="weights", param_type="choice", default="uniform", options=["uniform", "distance"], description="Weight function used in prediction"),
                    "metric": HyperparameterSpec(name="metric", param_type="choice", default="minkowski", options=["minkowski", "euclidean", "manhattan"], description="Distance metric"),
                },
                default_hyperparameters={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
            ),
            KNeighborsClassifier,
            aliases=["knn-classifier", "knn"],
        )

        # 1.5 Support Vector Classifier (SVC)
        self.register(
            AlgorithmInfo(
                id="svc",
                display_name="Support Vector Classifier",
                task="classification",
                is_supervised=True,
                description="Maximum-margin hyperplane separation with kernel embeddings (RBF, Polynomial, Linear).",
                family="svm",
                tags=["Kernel", "Margin-based", "Non-linear"],
                hyperparameters={
                    "C": HyperparameterSpec(name="C", param_type="float", default=1.0, min_value=0.01, max_value=1000.0, description="Regularization parameter"),
                    "kernel": HyperparameterSpec(name="kernel", param_type="choice", default="rbf", options=["linear", "poly", "rbf", "sigmoid"], description="Kernel type used in the algorithm"),
                    "gamma": HyperparameterSpec(name="gamma", param_type="choice", default="scale", options=["scale", "auto"], description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"),
                    "degree": HyperparameterSpec(name="degree", param_type="int", default=3, min_value=1, max_value=10, description="Degree of the polynomial kernel function ('poly')"),
                    "probability": HyperparameterSpec(name="probability", param_type="bool", default=True, description="Enable probability estimates"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"C": 1.0, "kernel": "rbf", "gamma": "scale", "degree": 3, "probability": True, "random_state": 42},
            ),
            SVC,
            aliases=["support-vector-machine", "svm", "svc-classifier"],
        )

        # 1.6 Gaussian Naive Bayes
        self.register(
            AlgorithmInfo(
                id="gaussian_nb",
                display_name="Gaussian Naive Bayes",
                task="classification",
                is_supervised=True,
                description="Bayes rule probabilistic classifier assuming feature conditional independence under Gaussian distributions.",
                family="bayes",
                tags=["Probabilistic", "Fast", "Baseline"],
                hyperparameters={
                    "var_smoothing": HyperparameterSpec(name="var_smoothing", param_type="float", default=1e-9, min_value=1e-15, max_value=1.0, description="Portion of largest variance added to variances for calculation stability"),
                },
                default_hyperparameters={"var_smoothing": 1e-9},
            ),
            GaussianNB,
            aliases=["naive-bayes", "gaussian-nb", "gnb"],
        )

        # 1.7 Gradient Boosting Classifier
        self.register(
            AlgorithmInfo(
                id="gradient_boosting_classifier",
                display_name="Gradient Boosting Classifier",
                task="classification",
                is_supervised=True,
                description="Sequential additive ensemble minimizing arbitrary differentiable loss functions.",
                family="ensemble",
                tags=["Boosting", "Ensemble", "State-of-the-Art"],
                hyperparameters={
                    "n_estimators": HyperparameterSpec(name="n_estimators", param_type="int", default=100, min_value=10, max_value=1000, description="Number of boosting stages to perform"),
                    "learning_rate": HyperparameterSpec(name="learning_rate", param_type="float", default=0.1, min_value=0.001, max_value=1.0, description="Shrinkage parameter scaling the contribution of each tree"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=3, min_value=1, max_value=20, description="Maximum depth of the individual regression estimators"),
                    "subsample": HyperparameterSpec(name="subsample", param_type="float", default=1.0, min_value=0.1, max_value=1.0, description="Fraction of samples used for fitting the individual base learners"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 42},
            ),
            GradientBoostingClassifier,
            aliases=["gradient-boosting-classifier", "gradient-boosting", "gbm"],
        )

        # ===================================================================
        # 2. REGRESSION ALGORITHMS (8)
        # ===================================================================

        # 2.1 Linear Regression
        self.register(
            AlgorithmInfo(
                id="linear_regression",
                display_name="Linear Regression",
                task="regression",
                is_supervised=True,
                description="Ordinary least squares linear regression estimating continuous target parameters.",
                family="linear",
                tags=["Linear", "OLS", "Analytical", "Interpretable"],
                hyperparameters={
                    "fit_intercept": HyperparameterSpec(name="fit_intercept", param_type="bool", default=True, description="Whether to calculate the intercept for this model"),
                },
                default_hyperparameters={"fit_intercept": True},
            ),
            LinearRegression,
            aliases=["linear-regression", "ols"],
        )

        # 2.2 Ridge Regression
        self.register(
            AlgorithmInfo(
                id="ridge_regression",
                display_name="Ridge Regression",
                task="regression",
                is_supervised=True,
                description="Linear least squares with L2 Tikhonov regularization preventing coefficient inflation.",
                family="linear",
                tags=["Linear", "L2", "Regularized"],
                hyperparameters={
                    "alpha": HyperparameterSpec(name="alpha", param_type="float", default=1.0, min_value=0.0001, max_value=10000.0, description="Regularization strength; must be a positive float"),
                    "fit_intercept": HyperparameterSpec(name="fit_intercept", param_type="bool", default=True, description="Calculate intercept term"),
                    "solver": HyperparameterSpec(name="solver", param_type="choice", default="auto", options=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], description="Solver algorithm"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"alpha": 1.0, "fit_intercept": True, "solver": "auto", "random_state": 42},
            ),
            Ridge,
            aliases=["ridge", "ridge-regression"],
        )

        # 2.3 Lasso Regression
        self.register(
            AlgorithmInfo(
                id="lasso_regression",
                display_name="Lasso Regression",
                task="regression",
                is_supervised=True,
                description="Linear model with L1 regularization producing sparse parameter vectors (automatic feature selection).",
                family="linear",
                tags=["Linear", "L1", "Sparse", "Feature Selection"],
                hyperparameters={
                    "alpha": HyperparameterSpec(name="alpha", param_type="float", default=1.0, min_value=0.0001, max_value=10000.0, description="Constant that multiplies the L1 term"),
                    "fit_intercept": HyperparameterSpec(name="fit_intercept", param_type="bool", default=True, description="Calculate intercept"),
                    "max_iter": HyperparameterSpec(name="max_iter", param_type="int", default=1000, min_value=10, max_value=10000, description="Maximum number of iterations"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"alpha": 1.0, "fit_intercept": True, "max_iter": 1000, "random_state": 42},
            ),
            Lasso,
            aliases=["lasso", "lasso-regression"],
        )

        # 2.4 Decision Tree Regressor
        self.register(
            AlgorithmInfo(
                id="decision_tree_regressor",
                display_name="Decision Tree Regressor",
                task="regression",
                is_supervised=True,
                description="Non-parametric hierarchical regression splitting variance over local feature regions.",
                family="tree",
                tags=["Tree", "Non-linear", "Interpretable"],
                hyperparameters={
                    "criterion": HyperparameterSpec(name="criterion", param_type="choice", default="squared_error", options=["squared_error", "friedman_mse", "absolute_error", "poisson"], description="Split evaluation criterion"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=None, min_value=1, max_value=100, allow_none=True, description="Maximum tree depth"),
                    "min_samples_split": HyperparameterSpec(name="min_samples_split", param_type="int", default=2, min_value=2, max_value=50, description="Minimum samples to split node"),
                    "min_samples_leaf": HyperparameterSpec(name="min_samples_leaf", param_type="int", default=1, min_value=1, max_value=50, description="Minimum samples at leaf"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"criterion": "squared_error", "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42},
            ),
            DecisionTreeRegressor,
            aliases=["decision-tree-regressor", "decision-tree", "dt-regressor"],
        )

        # 2.5 Random Forest Regressor
        self.register(
            AlgorithmInfo(
                id="random_forest_regressor",
                display_name="Random Forest Regressor",
                task="regression",
                is_supervised=True,
                description="Bagged ensemble of randomized regression trees averaging continuous outputs to minimize variance.",
                family="ensemble",
                tags=["Ensemble", "Bagging", "Non-linear", "Robust"],
                hyperparameters={
                    "n_estimators": HyperparameterSpec(name="n_estimators", param_type="int", default=100, min_value=10, max_value=1000, description="Number of trees"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=None, min_value=1, max_value=100, allow_none=True, description="Maximum tree depth"),
                    "min_samples_split": HyperparameterSpec(name="min_samples_split", param_type="int", default=2, min_value=2, max_value=50, description="Minimum samples to split node"),
                    "min_samples_leaf": HyperparameterSpec(name="min_samples_leaf", param_type="int", default=1, min_value=1, max_value=50, description="Minimum samples at leaf"),
                    "max_features": HyperparameterSpec(name="max_features", param_type="choice", default="sqrt", options=["sqrt", "log2", None], allow_none=True, description="Features evaluated per split"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42},
            ),
            RandomForestRegressor,
            aliases=["random-forest-regressor", "random-forest", "rf-regressor"],
        )

        # 2.6 K-Nearest Neighbors Regressor
        self.register(
            AlgorithmInfo(
                id="knn_regressor",
                display_name="K-Nearest Neighbors Regressor",
                task="regression",
                is_supervised=True,
                description="Continuous target interpolation based on local neighborhood point averaging.",
                family="neighbors",
                tags=["Instance-based", "Non-parametric", "Metric"],
                hyperparameters={
                    "n_neighbors": HyperparameterSpec(name="n_neighbors", param_type="int", default=5, min_value=1, max_value=100, description="Number of neighbors"),
                    "weights": HyperparameterSpec(name="weights", param_type="choice", default="uniform", options=["uniform", "distance"], description="Weighting strategy"),
                    "metric": HyperparameterSpec(name="metric", param_type="choice", default="minkowski", options=["minkowski", "euclidean", "manhattan"], description="Distance metric"),
                },
                default_hyperparameters={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
            ),
            KNeighborsRegressor,
            aliases=["knn-regressor", "knn"],
        )

        # 2.7 Support Vector Regressor (SVR)
        self.register(
            AlgorithmInfo(
                id="svr",
                display_name="Support Vector Regressor",
                task="regression",
                is_supervised=True,
                description="Epsilon-insensitive loss tube maximum-margin regression with non-linear kernel transformations.",
                family="svm",
                tags=["Kernel", "Margin-based", "Robust"],
                hyperparameters={
                    "C": HyperparameterSpec(name="C", param_type="float", default=1.0, min_value=0.01, max_value=1000.0, description="Regularization parameter"),
                    "epsilon": HyperparameterSpec(name="epsilon", param_type="float", default=0.1, min_value=0.0, max_value=10.0, description="Epsilon in the epsilon-SVR model tube"),
                    "kernel": HyperparameterSpec(name="kernel", param_type="choice", default="rbf", options=["linear", "poly", "rbf", "sigmoid"], description="Kernel type"),
                    "gamma": HyperparameterSpec(name="gamma", param_type="choice", default="scale", options=["scale", "auto"], description="Kernel coefficient"),
                },
                default_hyperparameters={"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"},
            ),
            SVR,
            aliases=["support-vector-regressor", "svr-regressor", "svm"],
        )

        # 2.8 Gradient Boosting Regressor
        self.register(
            AlgorithmInfo(
                id="gradient_boosting_regressor",
                display_name="Gradient Boosting Regressor",
                task="regression",
                is_supervised=True,
                description="Additive forward stage-wise gradient boosting of regression trees on least squares residual errors.",
                family="ensemble",
                tags=["Boosting", "Ensemble", "State-of-the-Art"],
                hyperparameters={
                    "n_estimators": HyperparameterSpec(name="n_estimators", param_type="int", default=100, min_value=10, max_value=1000, description="Boosting stages"),
                    "learning_rate": HyperparameterSpec(name="learning_rate", param_type="float", default=0.1, min_value=0.001, max_value=1.0, description="Step size shrinkage factor"),
                    "max_depth": HyperparameterSpec(name="max_depth", param_type="int", default=3, min_value=1, max_value=20, description="Individual tree depth"),
                    "subsample": HyperparameterSpec(name="subsample", param_type="float", default=1.0, min_value=0.1, max_value=1.0, description="Sample fraction per tree"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 42},
            ),
            GradientBoostingRegressor,
            aliases=["gradient-boosting-regressor", "gradient-boosting", "gbr"],
        )


        # ===================================================================
        # 3. CLUSTERING ALGORITHMS (3) (Unsupervised Workflow)
        # ===================================================================

        # 3.1 K-Means Clustering
        self.register(
            AlgorithmInfo(
                id="kmeans",
                display_name="K-Means Clustering",
                task="clustering",
                is_supervised=False,
                description="Centroid-based vector quantization partitioning data into K distinct Voronoi cells.",
                family="clustering",
                tags=["Centroid", "Partitioning", "Unsupervised"],
                hyperparameters={
                    "n_clusters": HyperparameterSpec(name="n_clusters", param_type="int", default=3, min_value=2, max_value=50, description="Number of clusters to form as well as centroids to generate"),
                    "init": HyperparameterSpec(name="init", param_type="choice", default="k-means++", options=["k-means++", "random"], description="Method for initialization"),
                    "max_iter": HyperparameterSpec(name="max_iter", param_type="int", default=300, min_value=10, max_value=2000, description="Maximum number of iterations of the k-means algorithm"),
                    "random_state": HyperparameterSpec(name="random_state", param_type="int", default=42, description="Random seed"),
                },
                default_hyperparameters={"n_clusters": 3, "init": "k-means++", "max_iter": 300, "random_state": 42},
            ),
            KMeans,
            aliases=["k-means", "k-means-clustering"],
        )

        # 3.2 DBSCAN Clustering
        self.register(
            AlgorithmInfo(
                id="dbscan",
                display_name="DBSCAN Clustering",
                task="clustering",
                is_supervised=False,
                description="Density-Based Spatial Clustering of Applications with Noise; identifies dense core clusters and isolates arbitrary shape noise.",
                family="clustering",
                tags=["Density", "Noise-resilient", "Arbitrary-shape", "Unsupervised"],
                hyperparameters={
                    "eps": HyperparameterSpec(name="eps", param_type="float", default=0.5, min_value=0.01, max_value=50.0, description="Maximum distance between two samples for one to be considered as in the neighborhood of the other"),
                    "min_samples": HyperparameterSpec(name="min_samples", param_type="int", default=5, min_value=1, max_value=100, description="Number of samples in a neighborhood for a point to be considered as a core point"),
                    "metric": HyperparameterSpec(name="metric", param_type="choice", default="euclidean", options=["euclidean", "manhattan", "cosine"], description="Metric used when calculating distance between instances"),
                },
                default_hyperparameters={"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
            ),
            DBSCAN,
            aliases=["dbscan-clustering"],
        )

        # 3.3 Agglomerative Clustering
        self.register(
            AlgorithmInfo(
                id="agglomerative_clustering",
                display_name="Agglomerative Hierarchical Clustering",
                task="clustering",
                is_supervised=False,
                description="Bottom-up hierarchical clustering recursively merging pairs of clusters based on distance linkage.",
                family="clustering",
                tags=["Hierarchical", "Tree-based", "Dendrogram", "Unsupervised"],
                hyperparameters={
                    "n_clusters": HyperparameterSpec(name="n_clusters", param_type="int", default=3, min_value=2, max_value=50, description="Number of clusters to find"),
                    "linkage": HyperparameterSpec(name="linkage", param_type="choice", default="ward", options=["ward", "complete", "average", "single"], description="Linkage criterion determining distance between sets"),
                    "metric": HyperparameterSpec(name="metric", param_type="choice", default="euclidean", options=["euclidean", "manhattan", "cosine"], description="Metric used to compute linkage"),
                },
                default_hyperparameters={"n_clusters": 3, "linkage": "ward", "metric": "euclidean"},
            ),
            AgglomerativeClustering,
            aliases=["agglomerative", "hierarchical-clustering", "agglomerative-clustering"],
        )


_algorithm_registry: Optional[AlgorithmRegistry] = None


def get_algorithm_registry() -> AlgorithmRegistry:
    """Singleton getter for algorithm registry."""
    global _algorithm_registry
    if _algorithm_registry is None:
        _algorithm_registry = AlgorithmRegistry()
    return _algorithm_registry
