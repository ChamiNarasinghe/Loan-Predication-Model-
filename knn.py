import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from visualization import plot_feature_distributions, plot_class_distribution, plot_confusion_matrix, plot_roc_curve

# Load the dataset
df = pd.read_csv(r"E:\train.csv")

# Handle outliers for 'ApplicantIncome'
max_threshold_income = df['ApplicantIncome'].quantile(0.95)
min_threshold_income = df['ApplicantIncome'].quantile(0.05)
df = df[(df['ApplicantIncome'] >= min_threshold_income) & (df['ApplicantIncome'] <= max_threshold_income)]

# Handle outliers for 'LoanAmount'
max_threshold_loan = df['LoanAmount'].quantile(0.95)
min_threshold_loan = df['LoanAmount'].quantile(0.05)
df = df[(df['LoanAmount'] >= min_threshold_loan) & (df['LoanAmount'] <= min_threshold_loan)]

# Fill missing values for numeric columns
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median())

# Fill missing values for categorical columns with the mode
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
df['Loan_Status'] = df['Loan_Status'].fillna(df['Loan_Status'].mode()[0])

# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column + '_T'] = le.fit_transform(df[column])
    label_encoders[column] = le

# Convert 'Dependents' column to numeric
df['Dependents'] = df['Dependents'].replace({'3+': 3}).astype(int)

# Drop original categorical columns and other unnecessary columns
x = df.drop(columns=['Loan_Status'] + categorical_columns + ['Loan_ID'])
y = df['Loan_Status']

# Encode target variable
le_Loan_Status = LabelEncoder()
y = le_Loan_Status.fit_transform(y)

# Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=42)

# Train the model
model = KNeighborsClassifier(n_neighbors=19)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
y_scores = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_scores)

print("Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

# Save the model
joblib.dump(model, 'Loan_Prediction_Model.pkl')

# Plot distributions and performance metrics
plot_feature_distributions(df, ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome'])
plot_class_distribution(y_res)
plot_confusion_matrix(y_test, y_pred, labels=le_Loan_Status.classes_)
plot_roc_curve(y_test, y_scores)
