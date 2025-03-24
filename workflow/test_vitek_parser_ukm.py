# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import unittest
import pandas as pd
import vitek_parser_ukm
from constants import TRANSLATIONS_PATH


class TestVitekParserUkm(unittest.TestCase):
    def test_process(self):

        input_pd = pd.read_csv(
            "resources/test_data/UKM_VITEK_TestSet.csv", sep="\t", quotechar='"'
        )
        translation = pd.read_csv(TRANSLATIONS_PATH, sep=",")
        expected = pd.read_csv(
            "resources/test_output_control/vitek_parser_ukm/vitek_parser_ukm.csv",
            sep=",",
            quotechar='"',
        )

        actual = vitek_parser_ukm.process(input_pd, translation)

        actual = actual.replace("NA", pd.NA)
        expected = expected.replace("NA", pd.NA)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
