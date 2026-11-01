import pytest
from httpx import AsyncClient
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin

from services.ml.registry import get_algorithm_registry, AlgorithmRegistry
from core.exceptions import AppValidationError as ValidationError


def test_registry_contains_all_18_algorithms():
    registry = get_algorithm_registry()
    all_algos = registry.list_algorithms()
    assert len(all_algos) == 18

    class_algos = registry.list_algorithms(task="classification")
    assert len(class_algos) == 7

    regr_algos = registry.list_algorithms(task="regression")
    assert len(regr_algos) == 8

    clust_algos = registry.list_algorithms(task="clustering")
    assert len(clust_algos) == 3


def test_all_classification_algorithms_instantiate():
    registry = get_algorithm_registry()
    classification_ids = [
        "logistic_regression",
        "decision_tree_classifier",
        "random_forest_classifier",
        "knn_classifier",
        "svc",
        "gaussian_nb",
        "gradient_boosting_classifier",
    ]

    for algo_id in classification_ids:
        info = registry.get(algo_id)
        assert info.task == "classification"
        assert info.is_supervised is True

        # Test default instantiation
        estimator = registry.create_estimator(algo_id)
        assert isinstance(estimator, BaseEstimator)
        assert isinstance(estimator, ClassifierMixin)


def test_all_regression_algorithms_instantiate():
    registry = get_algorithm_registry()
    regression_ids = [
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "knn_regressor",
        "svr",
        "gradient_boosting_regressor",
    ]

    for algo_id in regression_ids:
        info = registry.get(algo_id)
        assert info.task == "regression"
        assert info.is_supervised is True

        # Test default instantiation
        estimator = registry.create_estimator(algo_id)
        assert isinstance(estimator, BaseEstimator)
        assert isinstance(estimator, RegressorMixin)


def test_all_clustering_algorithms_instantiate():
    registry = get_algorithm_registry()
    clustering_ids = [
        "kmeans",
        "dbscan",
        "agglomerative_clustering",
    ]

    for algo_id in clustering_ids:
        info = registry.get(algo_id)
        assert info.task == "clustering"
        assert info.is_supervised is False

        # Test default instantiation
        estimator = registry.create_estimator(algo_id)
        assert isinstance(estimator, BaseEstimator)
        assert isinstance(estimator, ClusterMixin)


def test_hyperparameter_configuration_and_custom_values():
    registry = get_algorithm_registry()

    # 1. Random Forest custom hyperparameters
    rf = registry.create_estimator(
        "random_forest_classifier",
        hyperparameters={"n_estimators": 50, "max_depth": 7, "min_samples_split": 4},
    )
    assert rf.n_estimators == 50
    assert rf.max_depth == 7
    assert rf.min_samples_split == 4

    # 2. Decision Tree custom hyperparameters
    dt = registry.create_estimator(
        "decision_tree_classifier",
        hyperparameters={"criterion": "entropy", "max_depth": 5},
    )
    assert dt.criterion == "entropy"
    assert dt.max_depth == 5

    # 3. Logistic Regression custom hyperparameters
    lr = registry.create_estimator(
        "logistic_regression",
        hyperparameters={"C": 0.5, "solver": "saga", "max_iter": 500},
    )
    assert lr.C == 0.5
    assert lr.solver == "saga"
    assert lr.max_iter == 500


def test_hyperparameter_validation_rejects_invalid_values():
    registry = get_algorithm_registry()

    # 1. Invalid option
    is_valid, _, errors = registry.validate_hyperparameters(
        "decision_tree_classifier",
        {"criterion": "invalid_criterion"},
    )
    assert is_valid is False
    assert any("Allowed options" in err for err in errors)

    # 2. Below minimum value
    is_valid, _, errors = registry.validate_hyperparameters(
        "random_forest_classifier",
        {"n_estimators": 2}, # min is 10
    )
    assert is_valid is False
    assert any("must be >= 10" in err for err in errors)

    # 3. Unknown parameter (arbitrary code injection prevention)
    is_valid, _, errors = registry.validate_hyperparameters(
        "linear_regression",
        {"arbitrary_payload": "__import__('os').system('ls')"},
    )
    assert is_valid is False
    assert any("Unknown parameter" in err for err in errors)


def test_algorithm_compatibility():
    registry = get_algorithm_registry()

    # Classification algorithm with regression task
    comp, errors = registry.validate_compatibility("random_forest_classifier", task="regression")
    assert comp is False
    assert any("REGRESSION" in err for err in errors)

    # Regression algorithm with classification task
    comp, errors = registry.validate_compatibility("linear_regression", task="classification")
    assert comp is False
    assert any("CLASSIFICATION" in err for err in errors)

    # Correct matching
    comp, errors = registry.validate_compatibility("random_forest_classifier", task="classification")
    assert comp is True
    assert len(errors) == 0


def test_alias_resolution_by_task():
    registry = get_algorithm_registry()

    # 'random-forest' for classification resolves to random_forest_classifier
    rf_class = registry.get("random-forest", task="classification")
    assert rf_class.id == "random_forest_classifier"

    # 'random-forest' for regression resolves to random_forest_regressor
    rf_regr = registry.get("random-forest", task="regression")
    assert rf_regr.id == "random_forest_regressor"

    # 'decision-tree' for classification resolves to decision_tree_classifier
    dt_class = registry.get("decision-tree", task="classification")
    assert dt_class.id == "decision_tree_classifier"

    # 'decision-tree' for regression resolves to decision_tree_regressor
    dt_regr = registry.get("decision-tree", task="regression")
    assert dt_regr.id == "decision_tree_regressor"


@pytest.mark.asyncio
async def test_algorithm_registry_api_endpoints(async_client: AsyncClient):
    # 1. List all algorithms
    list_res = await async_client.get("/api/v1/algorithms")
    assert list_res.status_code == 200
    all_json = list_res.json()
    assert all_json["total"] == 18

    # 2. Filter by classification
    class_res = await async_client.get("/api/v1/algorithms?task=classification")
    assert class_res.status_code == 200
    assert class_res.json()["total"] == 7

    # 3. Filter by regression
    regr_res = await async_client.get("/api/v1/algorithms?task=regression")
    assert regr_res.status_code == 200
    assert regr_res.json()["total"] == 8

    # 4. Filter by clustering
    clust_res = await async_client.get("/api/v1/algorithms?task=clustering")
    assert clust_res.status_code == 200
    assert clust_res.json()["total"] == 3

    # 5. Get algorithm details
    detail_res = await async_client.get("/api/v1/algorithms/random_forest_classifier")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["display_name"] == "Random Forest Classifier"
    assert "n_estimators" in detail["hyperparameters"]
    assert detail["default_hyperparameters"]["n_estimators"] == 100

    # 6. Validate Hyperparameters API
    val_res = await async_client.post(
        "/api/v1/algorithms/random_forest_classifier/validate",
        json={"hyperparameters": {"n_estimators": 150, "max_depth": 12}},
    )
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["valid"] is True
    assert val_data["sanitized_hyperparameters"]["n_estimators"] == 150
    assert val_data["sanitized_hyperparameters"]["max_depth"] == 12

    # 7. Validate Hyperparameters with invalid inputs
    bad_val_res = await async_client.post(
        "/api/v1/algorithms/random_forest_classifier/validate",
        json={"hyperparameters": {"n_estimators": 2, "unknown_field": "test"}},
    )
    assert bad_val_res.status_code == 200
    bad_data = bad_val_res.json()
    assert bad_data["valid"] is False
    assert len(bad_data["errors"]) >= 2

    # 8. Check Compatibility API
    comp_res = await async_client.post(
        "/api/v1/algorithms/random_forest_classifier/compatibility",
        json={"task": "classification"},
    )
    assert comp_res.status_code == 200
    assert comp_res.json()["compatible"] is True

    bad_comp_res = await async_client.post(
        "/api/v1/algorithms/random_forest_classifier/compatibility",
        json={"task": "regression"},
    )
    assert bad_comp_res.status_code == 200
    assert bad_comp_res.json()["compatible"] is False
