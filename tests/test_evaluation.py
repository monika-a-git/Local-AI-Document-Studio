"""
Tests for Evaluation module: indexing metrics, source faithfulness, and benchmark runner.
"""

import unittest
from pathlib import Path

from src.evaluation import (
    compute_source_faithfulness,
    evaluate_indexing_performance,
    run_benchmark_suite,
)


class TestEvaluation(unittest.TestCase):

    def test_compute_source_faithfulness(self):
        context = "The Arduino Uno is a microcontroller board based on the ATmega328P with 14 digital pins."
        # Grounded answer: tokens match context
        grounded_answer = "Arduino Uno microcontroller ATmega328P pins."
        score_high = compute_source_faithfulness(grounded_answer, context)
        self.assertGreater(score_high, 0.7)

        # Hallucinated answer: tokens do not appear in context
        hallucinated = "Quantum entanglement superconductors teleportation."
        score_low = compute_source_faithfulness(hallucinated, context)
        self.assertEqual(score_low, 0.0)

    def test_indexing_performance_metrics(self):
        sample_path = Path("data/uploads/ARDINO UNO .PPT.pdf")
        if sample_path.exists():
            perf = evaluate_indexing_performance(str(sample_path))
            self.assertIn("filename", perf)
            self.assertIn("total_chunks", perf)
            self.assertIn("characters_per_second", perf)
            self.assertGreater(perf["characters_per_second"], 0)
            self.assertGreater(perf["total_chunks"], 0)


if __name__ == "__main__":
    unittest.main()

