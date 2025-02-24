import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import preprocessing

# Load data
data_loader = preprocessing.DataLoader()
# TODO: Paths shall be set via terminal
preprocessed_data = data_loader.get_preprocessed_data(
    "output/mic_interpretation.csv", "resources/genotype"
)

X = preprocessed_data.merged_input[preprocessed_data.feature_cols]
y = preprocessed_data.merged_input[preprocessed_data.target_cols]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get probability predictions (needed for ROC)
y_score = rf_model.predict_proba(X_test)
y_pred = rf_model.predict(X_test)
print(y_pred[:, 11])
# TODO Bring back accuracy

# TODO Calculate for all antibiotics
# TODO Calculate for R, I and S?
y_score_roc = np.array(y_score[11])[:, 0]
print(y_score_roc)
y_test_roc = np.char.strip(np.array(y_test.iloc[:, 11].astype(str), dtype=str))
print(y_test_roc)
fpr, tpr, thresholds = roc_curve(y_test_roc, y_score_roc, pos_label="R")
print(fpr)
print(tpr)
roc_auc = auc(fpr, tpr)

print(roc_auc)
display = RocCurveDisplay(
    fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
)
display.plot()
plt.show()
