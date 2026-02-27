import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("C:\\New folder\\Downloads\\Logapriya\\Logapriya\\fetal_health.csv")

X = df.drop('fetal_health', axis=1)
y = df['fetal_health'] - 1  

_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Model F1 Score:", f1_score(y_test, y_pred, average='weighted'))

w_df = pd.DataFrame([new_data], columns=X.columns)
new_scaled = scaler.transform(new_df)
prediction = rf_model.predict(new_scaled)[0]
label_map = {
    0: "1 - Normal",
    1: "2 - Suspect",
    2: "3 - Pathological"
}
print("\n🧠 Predicted Fetal Health Class:", label_map[prediction])
