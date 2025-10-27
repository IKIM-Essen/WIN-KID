import os
import logging
from statistics import median

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
from execution_modes import ExecutionMode
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    roc_curve,
    auc,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from constants import ID_COLUMN, ORGANISM_COLUMN, RESISTANCE_MAPPING


@dataclass
class ResultDTO:
    fpr: dict
    tpr: dict
    roc_auc: dict
    pr_auc: dict
    accuracy: float
    precision: dict
    recall: dict
    f1: dict
    vme: dict
    me: dict
    y_pred: np.ndarray
    y_score: np.ndarray
    feature_importance: dict


def load_dataset_paths(path_file):
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Settings file '{path_file}' not found.")

    path_df = pd.read_csv(path_file)

    if not {"DataSetName", "PathToCsv", "PathToGff"}.issubset(path_df.columns):
        raise ValueError(
            "Settings file must contain columns: DataSetName, PathToCsv, PathToGff"
        )

    return path_df


def validate_target_values(y_train, y_test, target_name):
    if (y_train == 0).any():
        raise ValueError(f"❌ 0 value found in y_train for target '{target_name}'")
    if (y_test == 0).any():
        raise ValueError(f"❌ 0 value found in y_test for target '{target_name}'")
    if y_train.isna().any():
        raise ValueError(f"❌ NaN value found in y_train for target '{target_name}'")
    if y_test.isna().any():
        raise ValueError(f"❌ NaN value found in y_test for target '{target_name}'")


def generate_result(target_col, y_test_col, y_score_col, y_pred_col, feat_import):
    unique_classes = np.unique(y_test_col)

    fpr = {}
    tpr = {}
    roc_auc = {}
    pr_auc = {}
    precision = {}
    recall = {}
    f1 = {}
    vme = {}
    me = {}

    if config.EXECUTION_MODE == ExecutionMode.PREDICT_AND_SAVE:
        fpr, tpr, roc_auc, accuracy, precision, recall, f1, vme, me = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
    else:
        for label_class in unique_classes:
            y_test_binarized = (y_test_col == label_class).astype(int)
            y_pred_binarized = (y_pred_col == label_class).astype(int)

            if label_class >= y_score_col.shape[1]:
                logging.warning(
                    "Skipping class %s for %s, not in predictions",
                    label_class,
                    target_col,
                )
                continue

            fpr[label_class], tpr[label_class], _ = roc_curve(
                y_test_binarized, y_score_col[:, label_class]
            )
            roc_auc[label_class] = auc(fpr[label_class], tpr[label_class])
            pr_auc[label_class] = average_precision_score(
                y_test_binarized, y_score_col[:, label_class]
            )

            precision[label_class] = precision_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )
            recall[label_class] = recall_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )
            f1[label_class] = f1_score(
                y_test_binarized, y_pred_binarized, zero_division=0
            )

        accuracy = accuracy_score(y_test_col, y_pred_col)
        vme = calc_very_major_errors(y_test_col, y_pred_col)
        me = calc_major_errors(y_test_col, y_pred_col)

    return ResultDTO(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        vme=vme,
        me=me,
        y_pred=y_pred_col,
        y_score=y_score_col,
        feature_importance=feat_import,
    )


def calc_very_major_errors(y_test_col, y_pred_col):
    mask = (y_test_col == 3) & np.isin(y_pred_col, [1, 2])
    count = mask.sum()
    percentage_vme = count / len(y_test_col) * 100
    return percentage_vme


def calc_major_errors(y_test_col, y_pred_col):
    mask = (y_pred_col == 3) & np.isin(y_test_col, [1, 2])
    count = mask.sum()
    percentage_all = count / len(y_test_col) * 100
    return percentage_all


def evaluate_per_organism(y_pred, y_true, organism_codes, y_score, target_name):
    results = {}

    df = pd.DataFrame({"organism": organism_codes, "y_true": y_true, "y_pred": y_pred})

    for i in range(y_score.shape[1]):
        df[f"proba_class_{i}"] = y_score[:, i]

    for org_code, group in df.groupby("organism"):
        y_t = group["y_true"].values
        y_p = group["y_pred"].values

        proba_cols = [f"proba_class_{i}" for i in range(y_score.shape[1])]
        y_s = group[proba_cols].values

        # Skip if only 1 class is present (not valid for AUC)
        if len(np.unique(y_t)) < 2:
            logging.warning(
                "Skipping evaluation per organism for organism %s due to single class in target '%s'.",
                org_code,
                target_name,
            )

            roc_auc = pr_auc = np.nan
        else:
            result = generate_result("per_organism", y_t, y_s, y_p, feat_import=None)

            # Use mean of per-class metrics as representative score
            roc_auc = np.nanmean(list(result.roc_auc.values()))
            pr_auc = np.nanmean(list(result.pr_auc.values()))
            precision = np.nanmean(list(result.precision.values()))
            recall = np.nanmean(list(result.recall.values()))
            f1 = np.nanmean(list(result.f1.values()))
            acc = result.accuracy

            results[org_code] = {
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

    return results


def average_per_organism_results(per_fold_results_list):
    merged = {}

    for fold_result in per_fold_results_list:
        for org_code, metrics in fold_result.items():
            if org_code not in merged:
                merged[org_code] = defaultdict(list)
            for k, v in metrics.items():
                merged[org_code][k].append(v)

    averaged = {}
    for org_code, metrics in merged.items():
        averaged[org_code] = {k: np.nanmean(v_list) for k, v_list in metrics.items()}

    return averaged


def display_results(results_dto):
    reverse_mapping = {v: k for k, v in RESISTANCE_MAPPING.items()}
    with PdfPages("rf_roc_report.pdf") as pdf:
        for name in results_dto:
            result = results_dto[name]
            for class_label in result.roc_auc:
                if np.isnan(result.roc_auc[class_label]):
                    continue

                class_name = reverse_mapping.get(class_label, str(class_label))

                logging.debug(
                    "  Class %s ROC AUC: %s}",
                    class_name,
                    result.roc_auc[class_label],
                )

                display = RocCurveDisplay(
                    fpr=result.fpr[class_label],
                    tpr=result.tpr[class_label],
                    roc_auc=result.roc_auc[class_label],
                    estimator_name=f"RF-{class_name}",
                )
                fig = display.plot().figure_
                fig.suptitle(f"ROC - {name} (Class {class_name})")

                pdf.savefig(fig)
                plt.close(fig)


def evaluation_to_csv(results_dto_list, y_test_input_list, y_train_input_list):

    evaluation_df_list = []
    for idx, results_dto in enumerate(results_dto_list):
        y_test_input = y_test_input_list[idx]
        y_train_input = y_train_input_list[idx]
        evaluation_df = pd.DataFrame(
            columns=[
                "Accuracy",
                "ROC_Mean",
                "PR_Mean",
                "Precision_Mean",
                "Recall_Mean",
                "F1_Mean",
                "VME",
                "ME",
                "Test_Count_S",
                "Test_Count_I",
                "Test_Count_R",
                "Train_Count_S",
                "Train_Count_I",
                "Train_Count_R",
            ]
        )
        for _, name in enumerate(results_dto):
            result = results_dto[name]
            test_label_counts = y_test_input[name]
            train_label_counts = y_train_input[name]

            evaluation_df.loc[name] = [
                result.accuracy,
                np.nanmean(list(result.roc_auc.values())),
                np.nanmean(list(result.pr_auc.values())),
                np.nanmean(list(result.precision.values())),
                np.nanmean(list(result.recall.values())),
                np.nanmean(list(result.f1.values())),
                result.vme,
                result.me,
                test_label_counts.get(1, 0),
                test_label_counts.get(2, 0),
                test_label_counts.get(3, 0),
                train_label_counts.get(1, 0),
                train_label_counts.get(2, 0),
                train_label_counts.get(3, 0),
            ]

        evaluation_df.loc["Median"] = [
            median(evaluation_df["Accuracy"].dropna()),
            median(evaluation_df["ROC_Mean"].dropna()),
            median(evaluation_df["PR_Mean"].dropna()),
            median(evaluation_df["Precision_Mean"].dropna()),
            median(evaluation_df["Recall_Mean"].dropna()),
            median(evaluation_df["F1_Mean"].dropna()),
            median(evaluation_df["VME"].dropna()),
            median(evaluation_df["ME"].dropna()),
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ]
        evaluation_df_list.append(evaluation_df)

    evaluation_df_3d = np.array([df.values for df in evaluation_df_list])

    mean_array = np.nanmean(evaluation_df_3d, axis=0)
    std_array = np.nanstd(evaluation_df_3d, axis=0)

    mean_df = pd.DataFrame(
        mean_array,
        index=evaluation_df_list[0].index,
        columns=evaluation_df_list[0].columns,
    )
    std_df = pd.DataFrame(
        std_array,
        index=evaluation_df_list[0].index,
        columns=evaluation_df_list[0].columns,
    )

    print("Standard Deviation:")
    print(std_df)

    print("Mean Values:")
    print(mean_df)

    os.makedirs("Evaluation", exist_ok=True)

    mean_df.to_csv(
        "Evaluation/Evaluation.csv", index=True, header=True, index_label="Target"
    )
    std_df.to_csv(
        "Evaluation/Standard_Deviation.csv",
        index=True,
        header=True,
        index_label="Target",
    )


def per_organism_evaluation_to_csv(
    results_organism, output_path="Evaluation/Per_Organism_Evaluation.csv"
):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_metrics = set()
    rows = []
    for target, org_dict in results_organism.items():
        for organism, metrics in org_dict.items():
            row = {"Target": target, "Organism": organism}
            for metric_name, value in metrics.items():
                row[metric_name] = value
                all_metrics.add(metric_name)
            rows.append(row)

    df = pd.DataFrame(rows)

    for metric in all_metrics:
        metric_df = df.pivot(index="Target", columns="Organism", values=metric)
        if not metric_df.empty:
            metric_df.loc["Median"] = metric_df.median(numeric_only=True)
            metric_output_path = output_path.replace(".csv", f"_{metric}.csv")
            metric_df.to_csv(metric_output_path)
            logging.info("%s -table saved at: %s", metric.title(), metric_output_path)


def feature_importance_to_csv(results_dto_list):
    importance_df_list = []
    for results_dto in results_dto_list:
        importance_df = pd.DataFrame(columns=results_dto.keys())
        importance_df[ORGANISM_COLUMN] = None
        for name in results_dto:
            importance_df.loc[name] = 0.0

            def strip_prefix(name: str) -> str:
                prefixes = ("S_", "I_", "R_")
                for p in prefixes:
                    if name.startswith(p):
                        return name[len(p) :]
                return name

            result = results_dto[name]
            s_clean = result.feature_importance.rename(index=strip_prefix)
            s_clean = s_clean.groupby(s_clean.index).sum()
            importance_df.loc[name, s_clean.index] += s_clean
        importance_df_list.append(importance_df)
    os.makedirs("Evaluation", exist_ok=True)
    pd.concat(importance_df_list).groupby(level=0).mean().to_csv(
        "Evaluation/feature_importance.csv"
    )


def save_prediction_results(dataset_input, preprocessed_input, rf_results_input):
    pred_dict = {name: result.y_pred for name, result in rf_results_input[0].items()}
    pred_df = pd.DataFrame(pred_dict)

    inv_mapping = {v: k for k, v in RESISTANCE_MAPPING.items()}
    pred_df = pred_df.replace(inv_mapping)

    pred_df[ID_COLUMN] = preprocessed_input.merged_input[ID_COLUMN]
    cols = [ID_COLUMN] + [col for col in pred_df.columns if col != ID_COLUMN]
    pred_df = pred_df[cols]
    path = Path(dataset_input["PathToCsv"][0])
    new_path = path.parent / f"Result_{path.name}"
    pred_df.to_csv(new_path, index=False)
    print(pred_df)
