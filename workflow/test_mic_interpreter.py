# Copyright 2025 by Miriam Balzer & Julian Welling, University of Duisburg-Essen
# Licensed under the MIT License
# This file may be copied, modified, and distributed under the terms of the MIT License.

import unittest
import os
import pandas as pd
import mic_interpreter


class TestMicInterpreter(unittest.TestCase):
    def test_interpret_folder_ume(self):

        input_folder = "resources/test_output_control/vitek_parser_ume"
        expected_folder = "resources/test_output_control/mic_interpreter/UME"
        expected_dic = {}
        for expected_file_name in os.listdir(expected_folder):
            if expected_file_name.endswith(".csv"):
                expected_path = os.path.join(expected_folder, expected_file_name)
                expected_df = pd.read_csv(expected_path)
                expected_dic[expected_folder + "/" + expected_file_name] = expected_df

        actual_dic = mic_interpreter.interpret_folder(input_folder, expected_folder)

        self.assertEqual(len(expected_dic), len(actual_dic))

        for expected_key in expected_dic:
            expected = expected_dic[expected_key]
            actual = actual_dic[expected_key]
            actual = actual.replace("NA", pd.NA)
            expected = expected.replace("NA", pd.NA)
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)

    def test_interpret_folder_ukm(self):

        input_folder = "resources/test_output_control/vitek_parser_ukm"
        expected_folder = "resources/test_output_control/mic_interpreter/UKM"
        expected_dic = {}
        for expected_file_name in os.listdir(expected_folder):
            if expected_file_name.endswith(".csv"):
                expected_path = os.path.join(expected_folder, expected_file_name)
                expected_df = pd.read_csv(expected_path)
                expected_dic[expected_folder + "/" + expected_file_name] = expected_df

        actual_dic = mic_interpreter.interpret_folder(input_folder, expected_folder)

        self.assertEqual(len(expected_dic), len(actual_dic))

        for expected_key in expected_dic:
            expected = expected_dic[expected_key]
            actual = actual_dic[expected_key]
            actual = actual.replace("NA", pd.NA)
            expected = expected.replace("NA", pd.NA)
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)

    def test_interpret_folder_bvbcr(self):

        input_folder = "resources/test_output_control/vitek_parser_bvbcr"
        expected_folder = "resources/test_output_control/mic_interpreter/BVBCR"
        expected_dic = {}
        for expected_file_name in os.listdir(expected_folder):
            if expected_file_name.endswith(".csv"):
                expected_path = os.path.join(expected_folder, expected_file_name)
                expected_df = pd.read_csv(expected_path)
                expected_dic[expected_folder + "/" + expected_file_name] = expected_df

        actual_dic = mic_interpreter.interpret_folder(input_folder, expected_folder)

        self.assertEqual(len(expected_dic), len(actual_dic))

        for expected_key in expected_dic:
            expected = expected_dic[expected_key]
            actual = actual_dic[expected_key]
            actual = actual.replace("NA", pd.NA)
            expected = expected.replace("NA", pd.NA)
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
