"""
ZEMOO Machine Learning Module
Core foundation for model training, preprocessing, evaluation, and inference.
"""

from .analyzer import DatasetAnalyzer, get_dataset_analyzer
from .loader import DatasetLoader, get_dataset_loader
from .preprocessor import (
    OperationSummary,
    OutlierClipper,
    PreprocessedDataBundle,
    PreprocessingConfig,
    PreprocessingPipelineBuilder,
    PreprocessingPreviewResponse,
    SafeLogTransformer,
)
from .registry import (
    AlgorithmInfo,
    AlgorithmRegistry,
    HyperparameterSpec,
    get_algorithm_registry,
)
from .visualizer import VisualizationEngine, get_visualization_engine
from .pipeline_builder import DynamicPipelineBuilder, TrainingConfig, PipelineBuildResult
from .trainer import MLTrainingEngine, TrainingResult
from .job_manager import TrainingJobManager, ActiveJobState

__all__ = [
    "DatasetLoader",
    "get_dataset_loader",
    "DatasetAnalyzer",
    "get_dataset_analyzer",
    "VisualizationEngine",
    "get_visualization_engine",
    "PreprocessingPipelineBuilder",
    "PreprocessingConfig",
    "PreprocessingPreviewResponse",
    "PreprocessedDataBundle",
    "OperationSummary",
    "OutlierClipper",
    "SafeLogTransformer",
    "AlgorithmRegistry",
    "get_algorithm_registry",
    "AlgorithmInfo",
    "HyperparameterSpec",
    "DynamicPipelineBuilder",
    "TrainingConfig",
    "PipelineBuildResult",
    "MLTrainingEngine",
    "TrainingResult",
    "TrainingJobManager",
    "ActiveJobState",
]



