# Suppress warnings to keep output clean
import warnings
warnings.filterwarnings("ignore")

# Libraries for data manipulation
import pandas as pd
import numpy as np
import io

# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for model building and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, recall_score,
                            precision_score, f1_score, roc_auc_score, roc_curve)
from sklearn import metrics, tree

# For Colab file upload functionality
from google.colab import files

# Upload file via Colab interface
uploaded = files.upload()

# Automatically determine the file name
file_name = list(uploaded.keys())[0]

# Read the file into DataFrame
data = pd.read_csv(io.BytesIO(uploaded[file_name]))

# Examine data structure
print("File successfully loaded:", file_name)
print("\nFirst 5 rows:")
print(data.head())

# Get overall information about the dataset
print("\nDataset Information:")
data.info()

# Display basic statistics for numerical columns
print("\nDescriptive statistics for numerical columns:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Check if ID is unique and drop it
if 'ID' in data.columns:
    print(f"\nNumber of unique IDs: {data['ID'].nunique()} out of {len(data)} rows")
    # Drop ID column as it's not useful for modeling
    data.drop(["ID"], axis=1, inplace=True)
    print("ID column has been dropped.")

# Target variable distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='status', data=data, palette='viridis')
plt.title('Distribution of Target Variable (Status)', fontsize=15)
plt.xlabel('Status (0 = Not Converted, 1 = Converted)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add annotations
for p in ax.patches:
    height = p.get_height()
    percentage = 100 * height / len(data)
    ax.text(p.get_x() + p.get_width()/2., height + 50,
            f'{int(height)}\n({percentage:.1f}%)',
            ha="center", fontsize=12)

plt.show()

# Analyze numerical features
for col in [c for c in numerical_cols if c != 'status']:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution
    sns.histplot(data=data, x=col, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {col}')
    ax1.text(0.95, 0.95, f'Skew: {data[col].skew():.2f}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Boxplot by target
    sns.boxplot(x='status', y=col, data=data, ax=ax2)
    ax2.set_title(f'{col} by Conversion Status')
    ax2.set_xlabel('Status (0 = Not Converted, 1 = Converted)')

    plt.tight_layout()
    plt.show()

    # Print summary statistics by target
    print(f"Summary statistics for {col} by status:")
    print(data.groupby('status')[col].describe())
    print("\n")

# Analyze categorical features
for col in categorical_cols:
    if col != 'ID':
        # Display distribution
        plt.figure(figsize=(12, 5))
        ax = sns.countplot(x=col, hue='status', data=data, palette='Set2')
        plt.title(f'Distribution of {col} by Status', fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Status', labels=['Not Converted', 'Converted'])
        plt.tight_layout()
        plt.show()

        # Calculate conversion rate for each category
        conversion_rates = data.groupby(col)['status'].mean() * 100
        print(f"Conversion rates for {col}:")
        for category, rate in conversion_rates.items():
            print(f"- {category}: {rate:.2f}%")
        print("\n")

# Correlation analysis for numerical features
numeric_data = data[numerical_cols]
corr_matrix = numeric_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Correlation Matrix of Numerical Features', fontsize=15)
plt.tight_layout()
plt.show()

# Analyze relationship between categorical variables and conversion
print("\nKey observations from categorical variables:")
for col in categorical_cols:
    if col != 'ID':
        # Chi-square test for independence
        from scipy.stats import chi2_contingency
        contingency = pd.crosstab(data[col], data['status'])
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"{col} - Chi2: {chi2:.2f}, p-value: {p:.4f}")

# Create a copy for feature engineering
data_fe = data.copy()

# Age groups
data_fe['age_group'] = pd.cut(data_fe['age'],
                            bins=[0, 25, 35, 45, 55, 100],
                            labels=['<25', '25-35', '35-45', '45-55', '55+'])

# Website engagement score
from sklearn.preprocessing import MinMaxScaler

# Normalize engagement metrics
engagement_features = ['website_visits', 'time_spent_on_website', 'page_views_per_visit']
scaler = MinMaxScaler()
data_fe[engagement_features] = scaler.fit_transform(data_fe[engagement_features])

# Create engagement score (weighted average)
data_fe['engagement_score'] = (
    data_fe['website_visits'] * 0.3 +
    data_fe['time_spent_on_website'] * 0.4 +
    data_fe['page_views_per_visit'] * 0.3
)

# Convert profile_completed to numerical
profile_map = {'Low': 0, 'Medium': 1, 'High': 2}
data_fe['profile_completed_num'] = data_fe['profile_completed'].map(profile_map)

# Convert Yes/No to 1/0 for marketing channels
marketing_channels = ['print_media_type1', 'print_media_type2', 'digital_media',
                     'educational_channels', 'referral']

for channel in marketing_channels:
    data_fe[channel + '_binary'] = data_fe[channel].map({'Yes': 1, 'No': 0})

# Count total marketing channels per lead
data_fe['marketing_channel_count'] = data_fe[[c + '_binary' for c in marketing_channels]].sum(axis=1)

# Visualize new features
plt.figure(figsize=(10, 6))
sns.histplot(data=data_fe, x='engagement_score', hue='status', kde=True)
plt.title('Distribution of Engagement Score by Status', fontsize=15)
plt.xlabel('Engagement Score', fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='marketing_channel_count', y='status', data=data_fe, estimator=lambda x: np.mean(x) * 100)
plt.title('Conversion Rate by Marketing Channel Count', fontsize=15)
plt.xlabel('Number of Marketing Channels', fontsize=12)
plt.ylabel('Conversion Rate (%)', fontsize=12)
plt.show()

# Choose dataset for modeling
final_data = data_fe.copy()

# Drop original columns that have been transformed
columns_to_drop = ['profile_completed'] + marketing_channels
X = final_data.drop(['status'] + columns_to_drop, axis=1)
y = final_data['status']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

print("Final feature set:")
print(f"Number of features: {X.shape[1]}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Check class distribution
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True) * 100)
print("\nClass distribution in testing set:")
print(y_test.value_counts(normalize=True) * 100)

# Function to evaluate and visualize model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate a model's performance with detailed metrics and visualizations
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probability predictions for ROC curve
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_precision = metrics.precision_score(y_test, y_test_pred)
    test_recall = metrics.recall_score(y_test, y_test_pred)
    test_f1 = metrics.f1_score(y_test, y_test_pred)
    test_roc_auc = metrics.roc_auc_score(y_test, y_test_prob)

    # Print model performance
    print(f"===== {model_name} Performance =====")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"ROC AUC: {test_roc_auc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Converted', 'Converted'],
                yticklabels=['Not Converted', 'Converted'], ax=ax1)
    ax1.set_title(f'{model_name}: Confusion Matrix', fontsize=14)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_xlabel('Predicted', fontsize=12)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    ax2.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {test_roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_title(f'{model_name}: ROC Curve', fontsize=14)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Return metrics for comparison
    return {
        'model_name': model_name,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'roc_auc': test_roc_auc
    }

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train and evaluate each model
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    model_results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    results.append(model_results)
    print("-" * 50)

# Create a DataFrame with results for comparison
results_df = pd.DataFrame(results)
results_df = results_df.set_index('model_name')

# Display results sorted by F1 score
print("\nModel Performance Comparison (sorted by F1 Score):")
print(results_df.sort_values('f1', ascending=False))

# Visualize model comparison
plt.figure(figsize=(12, 6))
results_df.sort_values('f1', ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison', fontsize=15)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# Identify the best performing model based on F1 score
best_model_name = results_df['f1'].idxmax()
print(f"The best performing model is: {best_model_name}")

# Define hyperparameter grid for the best model
if best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', None],
        'class_weight': [None, 'balanced']
    }
    best_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')

elif best_model_name == "Decision Tree":
    param_grid = {
        'max_depth': [3, 5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', {0: 0.3, 1: 0.7}]
    }
    best_model = DecisionTreeClassifier(random_state=42)

elif best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 9, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    best_model = RandomForestClassifier(random_state=42)

# Create a scorer focused on F1 score
scorer = metrics.make_scorer(metrics.f1_score)

# Perform grid search with cross-validation
print(f"Tuning hyperparameters for {best_model_name}...")
grid_search = GridSearchCV(
    best_model,
    param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print best parameters
print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Get the best model
best_tuned_model = grid_search.best_estimator_

# Evaluate the tuned model
tuned_results = evaluate_model(best_tuned_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")

# Get feature importance from the best model
if hasattr(best_tuned_model, 'feature_importances_'):
    # For tree-based models
    feature_importance = best_tuned_model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # Print feature importance
    print("\nTop 15 Feature Importance:")
    print(importance_df.head(15))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=15)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

    # If it's a decision tree, visualize the tree
    if best_model_name == "Decision Tree":
        plt.figure(figsize=(20, 10))
        tree.plot_tree(best_tuned_model, feature_names=list(feature_names),
                      filled=True, rounded=True, fontsize=8, max_depth=3)
        plt.title('Decision Tree Visualization (Limited to Depth 3)', fontsize=15)
        plt.tight_layout()
        plt.show()

elif hasattr(best_tuned_model, 'coef_'):
    # For linear models
    feature_importance = best_tuned_model.coef_[0]
    feature_names = X.columns

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': feature_importance
    }).sort_values('Coefficient', ascending=False)

    # Print feature importance
    print("\nTop 15 Feature Coefficients:")
    print(importance_df.head(15))

    # Plot feature importance
    plt.figure(figsize=(12, 8))

    # Sort by absolute value for visualization
    importance_df['abs_coef'] = abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values('abs_coef', ascending=False).head(15)

    sns.barplot(x='Coefficient', y='Feature', data=importance_df)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Top 15 Feature Coefficients - {best_model_name}', fontsize=15)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

print("\n===== CONCLUSION AND RECOMMENDATIONS =====")

# Summary of the best model
print(f"\nBest Model: {best_model_name}")
print(f"F1 Score: {tuned_results['f1']:.4f}")
print(f"Recall (Sensitivity): {tuned_results['recall']:.4f}")
print(f"Precision: {tuned_results['precision']:.4f}")
print(f"Accuracy: {tuned_results['accuracy']:.4f}")
print(f"ROC AUC: {tuned_results['roc_auc']:.4f}")

# Top factors influencing conversion
if 'importance_df' in locals():
    print("\nTop 5 factors influencing conversion:")
    for i, row in importance_df.head(5).iterrows():
        feature_name = row['Feature']
        if 'Coefficient' in importance_df.columns:
            value = row['Coefficient']
            direction = "positive" if value > 0 else "negative"
            print(f"- {feature_name} (has a {direction} impact)")
        else:
            value = row['Importance']
            print(f"- {feature_name} (importance: {value:.4f})")

# Business recommendations
print("\nBusiness Recommendations:")
print("1. Focus on increasing website engagement (visits, time spent, page views)")
print("2. Encourage profile completion as it correlates with higher conversion")
print("3. Target specific age groups with higher conversion potential")
print("4. Optimize marketing channel mix based on effectiveness")
print("5. Implement personalized follow-up for leads with high conversion probability")
