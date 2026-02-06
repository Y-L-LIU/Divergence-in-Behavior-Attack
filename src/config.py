# config.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass(frozen=True)
class Config:
    """Static configuration for the MIA experiment."""
    # Generation settings
    num_samples: int = 8
    vllm_batch_size: int = 16
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    max_tokens: int = 2048

    # Evaluation settings
    test_size: float = 0.30
    random_state: int = 2025
    target_fprs: Tuple[float, ...] = (0.001, 0.01, 0.05)
    score_tolerance: float = 1e-6

    # Feature names (base metrics)
    # base_feature_names: List[str] = field(default_factory=lambda: [
    #     "neg_log_likelihood", "zlib_entropy_ratio", "minkowski_0.2",
    #     "reverse_kl_div", "avg_entropy"
    # ])
    base_feature_names: List[str] = field(default_factory=lambda: [
        "neg_log_likelihood", "zlib_entropy_ratio",
        "reverse_kl_div", "minkowski_0.2","avg_entropy"
    ])
    

# Global instance to be imported by other modules
CONF = Config()