import io
from pathlib import Path
import pytest
import pandas as pd
from httpx import AsyncClient, ASGITransport

from main import app
from services.ml.analyzer import DatasetAnalyzer
from services.ml.visualizer import VisualizationEngine
from services.ml.loader import DatasetLoader


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": ["ID-001", "ID-002", "ID-003", "ID-004", "ID-005"],
        "age": [25, 30, 45, 35, 50],
        "salary": [50000.0, 65000.0, 120000.0, 80000.0, 140000.0],
        "active": [True, False, True, True, False],
        "department": ["Sales", "Engineering", "Engineering", "Marketing", "Sales"],
        "join_date": ["2020-01-15", "2021-03-20", "2019-11-05", "2022-07-01", "2018-09-12"],
        "notes": [None, "High performer", None, "Remote", "On leave"],
    })


def test_analyzer_column_detection(sample_df):
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze_dataset(sample_df)

    assert analysis["total_rows"] == 5
    assert analysis["total_columns"] == 7
    assert analysis["duplicate_rows"] == 0
    assert analysis["missing_cells_total"] == 2

    col_map = {c["name"]: c for c in analysis["columns"]}
    assert col_map["id"]["inferred_type"] == "identifier"
    assert col_map["age"]["inferred_type"] == "numeric"
    assert col_map["salary"]["inferred_type"] == "numeric"
    assert col_map["active"]["inferred_type"] == "boolean"
    assert col_map["department"]["inferred_type"] == "categorical"
    assert col_map["join_date"]["inferred_type"] == "datetime"

    # Numeric stats checks
    salary_stats = col_map["salary"]["stats"]
    assert salary_stats["min"] == 50000.0
    assert salary_stats["max"] == 140000.0
    assert salary_stats["median"] == 80000.0


def test_analyzer_type_override(sample_df):
    analyzer = DatasetAnalyzer()
    # Override id to categorical
    analysis = analyzer.analyze_dataset(sample_df, type_overrides={"id": "categorical"})
    col_map = {c["name"]: c for c in analysis["columns"]}
    assert col_map["id"]["inferred_type"] == "categorical"


def test_visualization_engine(sample_df):
    visualizer = VisualizationEngine()
    viz = visualizer.generate_all_visualizations(sample_df)

    assert "age" in viz["histograms"]
    assert len(viz["histograms"]["age"]["counts"]) == 8
    assert "salary" in viz["box_plots"]
    assert viz["box_plots"]["salary"]["median"] == 80000.0
    assert "department" in viz["categories"]
    assert len(viz["categories"]["department"]["categories"]) >= 3
    assert len(viz["missing_overview"]) == 7
    assert "age" in viz["correlation_matrix"]["columns"]
    assert len(viz["correlation_matrix"]["matrix"]) == 2  # age and salary
    assert viz["scatter_plot"]["x_col"] in ["age", "salary"]


@pytest.fixture
async def async_client():
    import tempfile
    import os
    from core.config import get_settings
    from core.database.config import get_database
    import core.database.config as db_config

    test_db_fd, test_db_path = tempfile.mkstemp(suffix=".db")
    os.close(test_db_fd)
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    os.environ["DATABASE_URL"] = test_db_url
    get_settings.cache_clear()
    db_config._db = None

    db = get_database()
    await db.create_tables()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    await db.drop_tables()
    Path(test_db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_dataset_analysis_api_endpoints(async_client):
    client = async_client
    # 1. Upload dataset
    csv_data = (
        b"cust_id,income,credit_score,churn,country\n"
        b"C1,45000,720,0,US\n"
        b"C2,85000,680,1,UK\n"
        b"C3,32000,590,1,US\n"
        b"C4,95000,750,0,DE\n"
        b"C5,62000,710,0,UK\n"
    )
    files = {"file": ("test_analysis_sample.csv", io.BytesIO(csv_data), "text/csv")}
    res = await client.post("/api/v1/datasets", files=files)
    assert res.status_code == 201, res.text
    dataset_id = res.json()["id"]

    # 2. Get Analysis
    analysis_res = await client.get(f"/api/v1/datasets/{dataset_id}/analysis")
    assert analysis_res.status_code == 200, analysis_res.text
    analysis = analysis_res.json()
    assert analysis["total_rows"] == 5
    assert analysis["total_columns"] == 5
    col_types = {c["name"]: c["inferred_type"] for c in analysis["columns"]}
    assert col_types["income"] == "numeric"
    assert col_types["churn"] == "boolean"
    assert col_types["country"] == "categorical"

    # 3. Get Visualizations
    viz_res = await client.get(f"/api/v1/datasets/{dataset_id}/visualizations")
    assert viz_res.status_code == 200, viz_res.text
    viz = viz_res.json()
    assert "income" in viz["histograms"]
    assert len(viz["correlation_matrix"]["columns"]) >= 2

    # 4. Column Type Override
    override_res = await client.post(
        f"/api/v1/datasets/{dataset_id}/columns/override",
        json={"columns": {"churn": "categorical"}},
    )
    assert override_res.status_code == 200, override_res.text
    updated = override_res.json()
    updated_types = {c["name"]: c["inferred_type"] for c in updated["columns"]}
    assert updated_types["churn"] == "categorical"

    # 5. Clean up
    del_res = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_res.status_code == 200
