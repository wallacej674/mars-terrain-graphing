import unittest

import config


class TestConfig(unittest.TestCase):
    def test_required_config_fields_exist(self):
        self.assertIsInstance(config.TILE_PATH, str)
        self.assertIsInstance(config.OUTPUT_DIR, str)
        self.assertIsInstance(config.N_CLUSTERS, int)
        self.assertIsInstance(config.MIN_REGION_SIZE, int)
        self.assertIsInstance(config.RANDOM_SEED, int)

    def test_optional_path_ids_are_none_or_int(self):
        self.assertIn(type(config.START_REGION_ID), (type(None), int))
        self.assertIn(type(config.GOAL_REGION_ID), (type(None), int))


if __name__ == "__main__":
    unittest.main()
