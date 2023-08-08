import logging
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

class ThermoStabilityEstimator:
    """Abstract parent class."""
    def __init__(self, args=None):
        self.run_history = []
        self.args = args

    def run(self, sequences: list[str], ids: list[str], **kwargs) -> Dict[str, float]:
        """Run the estimator on the specified sequences.
        
        Args:
            sequences: List of protein sequences.
            ids: List of corresponding IDs.
        
        Returns:
            Dict[str, float]: A dictionary map of ids to estimated thermal stability."""
        assert len(sequences) == len(ids)
        result = self._run(sequences, ids, **kwargs)
        self.run_history.append((sequences, ids, result))
        return result

    def _run(self, sequences: list[str], ids: list[str]) -> Dict[str, float]:
        """Private method to be implemented by child classes."""
        raise NotImplementedError()

