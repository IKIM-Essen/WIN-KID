# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import os
import unittest
import pandas as pd
import random_forest


class TestRandomForest(unittest.TestCase):
    def test_process(self):

        dataset_list = random_forest.load_dataset_paths(
            "resources/test_data/random_forest_TestSet/DataPaths_predict.csv"
        )

        random_forest.process(dataset_list)

        output_path = (
            "resources/test_data/random_forest_TestSet/Result_BVBCR_use_case.csv"
        )
        self.assertTrue(os.path.exists(output_path), f"{output_path} was not created")

        df = pd.read_csv(output_path)

        self.assertFalse(df.empty, "CSV file is empty")
        self.assertGreater(len(df.columns), 0, "CSV file has no columns")


if __name__ == "__main__":
    unittest.main()
