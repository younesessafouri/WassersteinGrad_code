"""
Py4CastXai Utils Package
========================

Main utilities for XAI experiments.
"""

# Main entry point
from .main_utils import (
    ExperimentConfig,
    ModelManager,
    DataManager,
    ExplainerManager,
    ExperimentRunner,
    run_experiment,
)

# Evaluation
from .evaluation import (
    MetricEvaluator,
    EvaluationRunner,
    MetricAggregator,
)

# # Comparison
# from .comparison import (
#     ExplainerComparator,
#     ComparisonRunner,
#     ModelComparator,
#     ExplanationAnalyzer,
# )

# Analysis
from .spatial_analysis import SpatialAnalyzer
from .pca_analysis import PCAAnalyzer
from .distribution import DistributionRunner
# Visualization
from .visualization import (
    AutoregressiveVisualizer,
    MultiLevelVisualizer,
)

__all__ = [
    # Main
    'ExperimentConfig',
    'ModelManager',
    'DataManager',
    'ExplainerManager',
    'ExperimentRunner',
    'run_experiment',
    
    # Evaluation
    'MetricEvaluator',
    'EvaluationRunner',
    'MetricAggregator',
    
    # Comparison
    'ExplainerComparator',
    'ComparisonRunner',
    'ModelComparator',
    'ExplanationAnalyzer',
    
    # Analysis
    'SpatialAnalyzer',
    'PCAAnalyzer',
    
    # Visualization
    'AutoregressiveVisualizer',
    'MultiLevelVisualizer',
]

