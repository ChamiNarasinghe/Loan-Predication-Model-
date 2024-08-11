import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"E:\train.csv")

# Handle missing values and preprocessing
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median())
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
df['Loan_Status'] = df['Loan_Status'].fillna(df['Loan_Status'].mode()[0])

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column + '_T'] = le.fit_transform(df[column])
    label_encoders[column] = le

df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(float).fillna(0).astype(int)

df_encoded = df.drop(columns=categorical_columns)
max_thresold = df['ApplicantIncome'].quantile(0.95)
min_thresold = df['ApplicantIncome'].quantile(0.05)
new = df[(df['ApplicantIncome'] >= min_thresold) & (df['ApplicantIncome'] <= max_thresold)]

x = new.drop('Loan_Status', axis='columns')
y = new['Loan_Status']

le_Loan_Status = LabelEncoder()
y_encoded = le_Loan_Status.fit_transform(y)
new.loc[:, 'Loan_Status_T'] = y_encoded
x_new = x.drop(['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_ID'], axis='columns')
y_new = new['Loan_Status_T']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.3, random_state=42)

# Evaluate SVM model with different C values
C_values = [0.01, 0.1, 1, 10, 100]
scores = []

for C in C_values:
    svm = SVC(C=C)
    svm.fit(x_train, y_train)
    score = svm.score(x_test, y_test)
    scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(C_values, scores, marker='o')
plt.xscale('log')
plt.xlabel('C value (log scale)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy for Different C Values')
plt.show()

# Choose the best C value and train the final model
best_C = C_values[scores.index(max(scores))]
print(f'The best C value is {best_C}')

model = SVC(class_weight='balanced', C=best_C, probability=True)  # Added probability=True for ROC-AUC calculation
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_scores = model.predict_proba(x_test)[:, 1]  # Get the probabilities for the positive class

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_scores)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

# Print detailed classification report
print(classification_report(y_test, y_pred))

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_scores)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
