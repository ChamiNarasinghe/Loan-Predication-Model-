import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load the dataset
df = pd.read_csv(r"E:\train.csv")

# Handle outliers for 'ApplicantIncome'
max_threshold_income = df['ApplicantIncome'].quantile(0.95)
min_threshold_income = df['ApplicantIncome'].quantile(0.05)
df = df[(df['ApplicantIncome'] >= min_threshold_income) & (df['ApplicantIncome'] <= max_threshold_income)]

# Handle outliers for 'LoanAmount'
max_threshold_loan = df['LoanAmount'].quantile(0.95)
min_threshold_loan = df['LoanAmount'].quantile(0.05)
df = df[(df['LoanAmount'] >= min_threshold_loan) & (df['LoanAmount'] <= max_threshold_loan)]

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

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
y_scores = model.predict_proba(x_test)[:, 1]

# Calculate Metrics
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

# Printing the Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Save the model
joblib.dump(model, 'Loan_Prediction_Model_DecisionTree.pkl')

# Visualization Functions
def plot_feature_distributions(df, features):
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

def plot_class_distribution(y):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y)
    plt.title('Distribution of Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# Plot distributions and performance metrics
plot_feature_distributions(df, ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome'])
plot_class_distribution(y_res)
plot_confusion_matrix(y_test, y_pred, labels=le_Loan_Status.classes_)
plot_roc_curve(y_test, y_scores)

# Scatter Plot for Actual vs Predicted Status
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='w', linewidth=0.5)
plt.xlabel('Actual Status')
plt.ylabel('Predicted Status')
plt.title('Scatter Plot of Actual vs. Predicted Loan Status')
plt.grid(True)
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.yticks([0, 1], ['Not Approved', 'Approved'])
plt.show()
