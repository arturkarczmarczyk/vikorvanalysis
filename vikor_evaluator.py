import numpy as np
from comparator.evaluators.abstract_evaluator import AbstractEvaluator
from pyrepo_mcda.mcda_methods import VIKOR


class VikorEvaluator(AbstractEvaluator):
    def __init__(self, v: float):
        self.v = v
        super().__init__()


    def compute(self, decision_matrix: np.ndarray, weights: np.ndarray, impacts: np.ndarray) -> np.ndarray:
        vikor = VIKOR(v=self.v)

        Qs = vikor(decision_matrix, weights, impacts)

        return Qs
