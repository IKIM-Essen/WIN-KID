# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import unittest
import pandas as pd
import random_forest


class TestRandomForest(unittest.TestCase):
    def test_process(self):

        dataset_list = random_forest.load_dataset_paths(
            "resources/settings/DataPaths_predict.csv"
        )

        random_forest.process(dataset_list)

        actual = pd.read_csv(
            "resources/DataSets/BVBRC_use_case/Result_BVBCR_use_case.csv"
        )
        expected = pd.read_csv(
            "resources/DataSets/BVBRC_use_case/Result_BVBCR_use_case_expected.csv"
        )
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
