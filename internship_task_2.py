import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Titanic dataset
titanic_df = pd.read_csv('D:/titanic.csv')

# Step 2: Quick look at the first few rows of the dataset
print("First few rows of the dataset:")
print(titanic_df.head())

# Step 3: Check for missing values
print("\nChecking for missing values:")
print(titanic_df.isnull().sum())

# Step 4: Data Cleaning

# Handling missing values
median_age = titanic_df['Age'].median()
titanic_df['Age'].fillna(median_age, inplace=True)

# Drop rows with missing 'Embarked' values
titanic_df.dropna(subset=['Embarked'], inplace=True)

# Drop 'Cabin' column due to high number of missing values
titanic_df.drop(columns=['Cabin'], inplace=True)

# Step 5: Remove Duplicate Values
print(f"\nNumber of duplicate rows: {titanic_df.duplicated().sum()}")
titanic_df.drop_duplicates(inplace=True)

# Step 6: Handle Inconsistent Values (if any)
# Example: Convert 'Sex' column to categorical type
titanic_df['Sex'] = titanic_df['Sex'].astype('category')

# Step 7: Handle Outliers (optional)

# Step 8: Exploratory Data Analysis (EDA)

# Pairplot to visualize relationships between numerical variables
print("\nPairplot to visualize relationships between numerical variables:")
sns.pairplot(titanic_df, hue='Survived', height=3)
plt.show()

# Bar plot of survival by Pclass (ticket class)
print("\nBar plot of survival by Pclass:")
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic_df, ci=None)
plt.title('Survival Rate by Ticket Class')
plt.show()

# Histogram of age distribution
print("\nHistogram of Age distribution:")
plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot of fare distribution by Pclass
print("\nBoxplot of Fare distribution by Pclass:")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=titanic_df)
plt.title('Fare Distribution by Ticket Class')
plt.show()

# Step 9: Correlation Matrix (numerical columns only)
print("\nCorrelation Matrix:")
numerical_cols = titanic_df.select_dtypes(include=['number']).columns
corr_matrix = titanic_df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
