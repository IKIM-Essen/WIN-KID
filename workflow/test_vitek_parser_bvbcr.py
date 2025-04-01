# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import unittest
import pandas as pd
import vitek_parser_bvbcr
from constants import NAMES_PATH
from constants import TRANSLATIONS_PATH


class TestVitekParserBvbcr(unittest.TestCase):
    def test_process(self):

        input_pd = pd.read_csv(
            "resources/test_data/BVBCR_VITEK_TestSet.csv", sep=";", quotechar='"'
        )
        translation = pd.read_csv(TRANSLATIONS_PATH, sep=",")
        names = pd.read_csv(NAMES_PATH, sep=",")
        expected = pd.read_csv(
            "resources/test_output_control/vitek_parser_bvbcr/vitek_parser_bvbcr.csv",
            sep=",",
            quotechar='"',
        )

        actual = vitek_parser_bvbcr.process_bvbcr(input_pd, names, translation)

        actual = actual.replace("NA", pd.NA)
        expected = expected.replace("NA", pd.NA)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
