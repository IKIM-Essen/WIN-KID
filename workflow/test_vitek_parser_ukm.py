import unittest
import pandas as pd
import vitek_parser_ukm


class TestVitekParserUkm(unittest.TestCase):
    def test_process(self):

        input_pd = pd.read_csv(
            "resources/test_data/UKM_VITEK_TestSet.csv", sep="\t", quotechar='"'
        )
        expected = pd.read_csv(
            "resources/test_output_control/vitek_parser_ukm/vitek_parser_ukm.csv",
            sep=",",
            quotechar='"',
        )

        actual = vitek_parser_ukm.process(input_pd)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
