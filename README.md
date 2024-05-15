
## Overview
This project aims to construct and evaluate a preprocessing and modeling pipeline to predict the income level based on the UCI Machine Learning Repository's Adult dataset.

## Data Source
Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). Extract the files into the following directory relative to the `src` directory:


## Data Description
We will use the `adult.data` file for this analysis. This file is comma-separated and contains the following columns without headers:

| Variable Name    | Role       | Type         | Demographic       | Description                 | Units | Missing Values |
|------------------|------------|--------------|-------------------|-----------------------------|-------|----------------|
| age              | Feature    | Integer      | Age               | N/A                         |       | No             |
| workclass        | Feature    | Categorical  | Income            | Various employment types    |       | Yes            |
| fnlwgt           | Feature    | Integer      | N/A               | N/A                         |       | No             |
| education        | Feature    | Categorical  | Education Level   | Various educational levels  |       | No             |
| education-num    | Feature    | Integer      | Education Level   | N/A                         |       | No             |
| marital-status   | Feature    | Categorical  | Other             | Various marital statuses    |       | No             |
| occupation       | Feature    | Categorical  | Other             | Various occupations         |       | Yes            |
| relationship     | Feature    | Categorical  | Other             | Various relationships       |       | No             |
| race             | Feature    | Categorical  | Race              | Various races               |       | No             |
| sex              | Feature    | Binary       | Sex               | Male, Female                |       | No             |
| capital-gain     | Feature    | Integer      | N/A               | N/A                         |       | No             |
| capital-loss     | Feature    | Integer      | N/A               | N/A                         |       | No             |
| hours-per-week   | Feature    | Integer      | N/A               | N/A                         |       | No             |
| native-country   | Feature    | Categorical  | Other             | Various countries           |       | Yes            |
| income           | Target     | Binary       | Income            | >50K, <=50K                 |       | No             |

## Objective
To predict the `income` variable using a robust preprocessing and modeling pipeline with cross-validation to evaluate the performance.

## Implementation
The `adult.data` file should be loaded using pandas. Example code to load data and preprocess:

```python
import pandas as pd

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

adult_dt = pd.read_csv('../data/adult/adult.data', header=None, names=columns)
adult_dt['income'] = adult_dt['income'].str.strip().map({'>50K': 1, '<=50K': 0})
```

## Preprocessing
Numerical Variables: Use KNN-based imputation and scale features with robust statistics.
Categorical Variables: Use the most frequent value for imputation and apply one-hot encoding.

## Model Pipeline
Create a model pipeline using a RandomForestClassifier with preprocessing steps defined in a ColumnTransformer.

## Cross-Validation
Evaluate the model using metrics such as log loss, ROC AUC, accuracy, and balanced accuracy across different folds.

## Results Interpretation
Discuss the model's performance and provide insights from the cross-validation results.

## Usage
Ensure the dataset is located as specified.
Run the preprocessing and modeling code in a Python environment with required libraries installed.

## References
Becker, Barry, and Ronny Kohavi. 1996. Adult dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
