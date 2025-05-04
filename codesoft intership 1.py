# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\Somu C\Downloads\Titanic-Dataset.csv")
print("Dataset Loaded. Shape:", df.shape)
print(df.head())

#  Exploratory Data Analysis 
print("\nMissing Values:\n", df.isnull().sum())

# Visualize survival counts
sns.countplot(data=df, x='Survived')
plt.title('Survival Counts')
plt.show(block=False); plt.pause(3); plt.close()

# Survival by gender
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Gender')
plt.show(block=False); plt.pause(3); plt.close()

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.show(block=False); plt.pause(3); plt.close()

#  Data Cleaning
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

#  Train-Test Split
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Survival by Passenger Class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show(block=False); plt.pause(3); plt.close()

# Age Distribution by Gender and Survival
sns.violinplot(data=df, x='Sex', y='Age', hue='Survived', split=True)
plt.title('Age Distribution by Gender and Survival')
plt.show(block=False); plt.pause(3); plt.close()

# Fare Distribution by Class and Survival
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived')
plt.title('Fare by Passenger Class and Survival')
plt.show(block=False); plt.pause(3); plt.close()

#  Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show(block=False); plt.pause(3); plt.close()

# Survival Distribution
survived_counts = df['Survived'].value_counts()
labels = ['Not Survived', 'Survived']
plt.pie(survived_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'green'])
plt.title('Survival Rate')
plt.axis('equal')
plt.show(block=False); plt.pause(3); plt.close()

