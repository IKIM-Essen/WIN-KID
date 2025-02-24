from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, accuracy_score
import preprocessing


@dataclass
class ResultDTO:

    fpr: np.ndarray
    tpr: np.ndarray
    roc_auc: float
    accuracy: float
    y_pred: np.ndarray
    y_score: np.ndarray


data_loader = preprocessing.DataLoader()
# TODO: Paths shall be set via terminal
preprocessed_data = data_loader.get_preprocessed_data(
    "output/mic_interpretation.csv", "resources/genotype"
)

X = preprocessed_data.merged_input[preprocessed_data.feature_cols]
y = preprocessed_data.merged_input[preprocessed_data.target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_score = rf_model.predict_proba(X_test)
y_pred = rf_model.predict(X_test)

# Generate results
results = {}
col_index = 0
for col in y_test.columns:
    y_score_roc = np.array(y_score[col_index])[:, 0]
    y_test_roc = np.char.strip(
        np.array(y_test.iloc[:, col_index].astype(str), dtype=str)
    )
    # TODO Calculate for R, I and S?
    fpr, tpr, thresholds = roc_curve(y_test_roc, y_score_roc, pos_label="R")
    roc_auc = auc(fpr, tpr)

    y_pred_df = pd.DataFrame(y_pred, columns=preprocessed_data.target_cols)
    accuracy = accuracy_score(y_test[col], y_pred_df[col])
    results[col] = ResultDTO(fpr, tpr, roc_auc, accuracy, y_pred_df[col], y_score_roc)
    col_index += 1

# Display results
for name in results:
    result = results[name]
    print(f"{name}    ROC AUC: {result.roc_auc}     Accuracy: {result.accuracy}")

    display = RocCurveDisplay(
        fpr=result.fpr, tpr=result.tpr, roc_auc=result.roc_auc, estimator_name="RF"
    )
    ax = display.plot().ax_
    ax.set_title(f"ROC - {name}")

plt.show()
