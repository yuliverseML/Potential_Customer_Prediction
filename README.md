# Potential_Customer_Prediction


## Overview

This project aims to predict the conversion of leads (potential customers) to paying customers for ExtraaLearn, an online education company. By analyzing lead behavior and characteristics, developed machine learning models to identify patterns that indicate a higher likelihood of conversion. This enables targeted marketing efforts and optimized resource allocation.

## Dataset (Marketing data for customer conversion prediction) [ExtraaLearn Customer Dataset](https://github.com/yuliverseML/Potential_Customer_Prediction/blob/main/ExtraaLearn.csv) 

The dataset contains 4,612 leads with 15 attributes describing their characteristics and interactions with ExtraaLearn:

- **ID**: Unique identifier for each lead
- **age**: Age of the lead
- **current_occupation**: Professional status (Professional/Unemployed/Student)
- **first_interaction**: How the lead first engaged with ExtraaLearn (Website/Mobile App)
- **profile_completed**: Profile completion percentage (Low/Medium/High)
- **website_visits**: Number of website visits
- **time_spent_on_website**: Total time spent on website (seconds)
- **page_views_per_visit**: Average number of pages viewed per visit
- **last_activity**: Last interaction type (Email/Phone/Website Activity)
- **Marketing channels**: Five binary fields indicating marketing exposure
  - print_media_type1
  - print_media_type2
  - digital_media
  - educational_channels
  - referral
- **status**: Target variable (1 = converted, 0 = not converted)

Class distribution shows an imbalanced dataset with approximately 29.9% conversion rate.

## Features

### Data Exploration

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of all variables to understand distributions and relationships
- **Target Distribution Analysis**: Examination of class balance and overall conversion rate
- **Correlation Analysis**: Identification of relationships between numerical features
- **Categorical Analysis**: Chi-square tests to determine significant categorical predictors
- **Behavioral Pattern Analysis**: Special focus on website engagement metrics as potential predictors

### Data Preprocessing

- **Handling Missing Values**: Verification of data completeness (no missing values found)
- **Feature Engineering**:
  - Age grouping into meaningful segments
  - Engagement score creation from website activity metrics
  - Binary conversion of marketing channel variables
  - Creation of marketing channel count metric
  - Numerical conversion of ordinal variables
- **Categorical Encoding**: One-hot encoding of categorical variables
- **Data Splitting**: 70/30 train-test split with stratification to maintain class balance

### Model Training

- **Algorithm Selection**: Implementation of three different algorithms to capture various data patterns:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- **Hyperparameter Tuning**: Grid search with cross-validation to optimize model parameters
  - For Decision Trees: depth, min samples, split criterion, class weights
  - For Random Forest: estimators, depth, sample parameters, class weights
  - For Logistic Regression: regularization strength, penalty type, class weights
- **Class Imbalance Handling**: Implementation of class weighting to address the imbalanced target distribution

### Model Evaluation

- **Performance Metrics**: Comprehensive evaluation using multiple metrics:
  - Accuracy: Overall correctness
  - Precision: Proportion of true conversions among predicted conversions
  - Recall: Proportion of actual conversions correctly identified
  - F1 Score: Harmonic mean of precision and recall
  - ROC AUC: Area under the ROC curve measuring discrimination ability
- **Cross-Validation**: 5-fold cross-validation to ensure model robustness
- **Confusion Matrix Analysis**: Detailed examination of prediction errors

### Visualization

- **Feature Distribution Plots**: Histograms and boxplots for numerical features
- **Categorical Relationship Charts**: Bar plots showing conversion rates by category
- **Correlation Heatmaps**: Visual representation of feature relationships
- **ROC Curves**: Visualization of model discrimination ability
- **Feature Importance Plots**: Bar charts highlighting influential features
- **Decision Tree Visualization**: Visual representation of decision rules (for Decision Tree model)

## Results

### Model Comparison

Performance metrics for all implemented models (sorted by F1 Score):

| Model               | Accuracy | Precision | Recall  | F1 Score | ROC AUC  |
|---------------------|----------|-----------|---------|----------|----------|
| Random Forest       | 0.844653 | 0.770492  | 0.682809| 0.724005 | 0.910267 |
| Decision Tree       | 0.795520 | 0.642544  | 0.709443| 0.674338 | 0.771795 |
| Logistic Regression | 0.815029 | 0.718663  | 0.624697| 0.668394 | 0.879665 |

### Best Model

After hyperparameter tuning, the Random Forest model achieved the following improved metrics:

- Accuracy: 0.8562
- Precision: 0.7388
- Recall: 0.8015
- F1 Score: 0.7689
- ROC AUC: 0.9200

### Feature Importance

Top 15 features driving the prediction (from Random Forest model):

| Feature                      | Importance |
|------------------------------|------------|
| first_interaction_Website    | 0.231090   |
| time_spent_on_website        | 0.209394   |
| profile_completed_num        | 0.141894   |
| engagement_score             | 0.126176   |
| age                          | 0.062172   |
| page_views_per_visit         | 0.057497   |
| website_visits               | 0.030618   |
| last_activity_Phone Activity | 0.030558   |
| current_occupation_Unemployed| 0.022117   |
| last_activity_Website Activity| 0.019614  |
| current_occupation_Student   | 0.019572   |
| marketing_channel_count      | 0.009782   |
| age_group_55+                | 0.006966   |
| referral_binary              | 0.005509   |
| age_group_45-55              | 0.004915   |

## Outcome

### Best Performing Model

**Random Forest** emerged as the best model with an F1 score of 0.7689, significantly outperforming other algorithms. Key findings:

1. Website interaction features are the strongest predictors of conversion
2. Engagement metrics (time spent, page views) strongly correlate with conversion likelihood
3. Profile completion level is a significant indicator of conversion intent
4. Age and occupation provide demographic signals for targeting
5. Multi-channel marketing exposure shows positive correlation with conversion

### Key Business Recommendations

1. **Enhance Website Experience**: Focus on increasing engagement metrics that strongly predict conversion
2. **Encourage Profile Completion**: Implement incentives or simplified forms to increase completion rates
3. **Age-Targeted Marketing**: Develop specialized campaigns for age segments with higher conversion potential
4. **Optimize Marketing Channels**: Allocate budget toward channels showing strongest correlation with conversion
5. **Implement Lead Scoring**: Deploy the model to prioritize high-probability leads for personalized follow-up

## Future Work

1. **Feature Engineering Refinement**: Develop more sophisticated engagement metrics
2. **Additional Models**: Explore gradient boosting algorithms (XGBoost, LightGBM) for potential performance improvement
3. **Temporal Analysis**: Incorporate time-series components to predict optimal contact timing
4. **Deep Learning**: Investigate neural network approaches for complex pattern recognition
5. **A/B Testing Framework**: Develop system to validate model-driven recommendations
6. **Deployment Pipeline**: Create an automated pipeline for model retraining and deployment

## Notes

- The model was trained on historical data and should be periodically retrained as customer behavior evolves
- While the Random Forest model provides good performance, real-world deployment should include regular monitoring for concept drift
- Feature importance can shift over time as market conditions change

## Contributing

Contributions to this project are welcome! Please follow these steps:


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
