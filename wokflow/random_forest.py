import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import preprocessing

data_loader = preprocessing.DataLoader()
merged_data = data_loader.get_merged_data()

target_cols = merged_data.columns[3:17]
feature_cols = merged_data.columns[17:]

X = merged_data[feature_cols]
y = merged_data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=target_cols)

for col in y_test.columns:
    accuracy = accuracy_score(y_test[col], y_pred_df[col])
    print(f"Accuracy for {col}: {accuracy}")
