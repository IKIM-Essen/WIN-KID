import unittest
import pandas as pd
import vitek_parser_ume


class TestVitekParserUkm(unittest.TestCase):
    def test_process(self):

        input_pd = pd.read_csv("resources/test_data/UME_VITEK_TestSet.csv", sep=",")
        translation = pd.read_csv("resources/settings/translations.csv", sep=",")
        names = pd.read_csv("resources/settings/names.csv", sep=",")
        expected_dir = "resources/test_output_control/vitek_parser_ume"

        actuals = vitek_parser_ume.process(input_pd, names, translation)

        for name in names["Vitek_Name"]:
            expected_path = (
                (expected_dir + "/" + name + ".csv").lower().replace(" ", "_")
            )
            expected = pd.read_csv(
                expected_path,
                sep=",",
            )
            actual = actuals[name]
            actual = actual.reset_index(drop=True)
            expected = expected.reset_index(drop=True)
            actual = actual.replace("NA", pd.NA)
            expected = expected.replace("NA", pd.NA)
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
