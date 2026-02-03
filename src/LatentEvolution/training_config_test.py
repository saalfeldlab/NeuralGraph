"""
test that ModelParams can round-trip through yaml.safe_load.
"""

import unittest
from pathlib import Path
import yaml

from LatentEvolution.eed_model import ModelParams


class TestModelParamsYamlRoundTrip(unittest.TestCase):
    """ensure model_dump() produces yaml.safe_load-compatible output."""

    def _load_default_config(self):
        default_path = Path(__file__).resolve().parent / "latent_1step.yaml"
        with open(default_path, "r") as f:
            data = yaml.safe_load(f)
        return ModelParams(**data)

    def test_safe_load_round_trip(self):
        cfg = self._load_default_config()
        dumped = yaml.dump(cfg.model_dump(mode='json'), sort_keys=False)
        loaded = yaml.safe_load(dumped)
        self.assertIsInstance(loaded, dict)
        self.assertIn("training", loaded)
        self.assertIn("stimulus_frequency", loaded["training"])


if __name__ == "__main__":
    unittest.main()
