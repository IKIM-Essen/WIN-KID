# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import unittest
import pandas as pd
import gff3_formater
from constants import GFF_COLUMNS


class TestGffFormatter(unittest.TestCase):
    def test_txt_to_gff3(self):

        input_df = pd.read_csv(
            "resources/test_data/CARD_results.txt",
            sep="\t",
            dtype=str,
        )
        expected_df = pd.read_csv(
            "resources/test_output_control/gff3_formater/CARD_result.gff",
            sep="\t",
            comment="#",
            names=GFF_COLUMNS,
            dtype=str,
        )

        actual_df = gff3_formater.txt_to_gff3(input_df)

        pd.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
