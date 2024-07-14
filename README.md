# Adult Income Prediction

This project aims to predict whether an individual's income exceeds $50,000 per year using the Adult Income dataset from Kaggle. The dataset includes various demographic features such as age, education, occupation, and more. The project involves data preprocessing, feature encoding, and building a classification model using RandomForestClassifier. Hyperparameter tuning is performed using GridSearchCV to optimize the model.

## Dataset

The dataset used in this project is available on Kaggle: [Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset).

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical features, and feature scaling.
- **Classification Model**: Building and training a RandomForestClassifier model.
- **Model Evaluation**: Evaluating the model using accuracy score and feature importance.
- **Hyperparameter Tuning**: Using GridSearchCV for optimizing the model.

## Requirements

- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/adult-income-prediction.git
    ```

2. **Navigate to the project directory**:
    ```sh
    cd adult-income-prediction
    ```

3. **Install the required packages**:
    ```sh
    pip install pandas seaborn matplotlib scikit-learn
    ```

## Usage

1. **Load the dataset**:
    ```python
    import pandas as pd
    df = pd.read_csv("income.csv")
    ```

2. **Data Preprocessing**:
    ```python
    df.education.value_counts()
    df.occupation.value_counts()
    df.workclass.value_counts()
    df.gender.value_counts()

    pd.get_dummies(df.occupation).add_prefix('occupation_')
    df = pd.concat([df.drop('occupation', axis=1), pd.get_dummies(df.occupation).add_prefix('occupation_')], axis=1)
    df = pd.concat([df.drop('workclass', axis=1), pd.get_dummies(df.workclass).add_prefix('workclass_')], axis=1)
    df = df.drop('education', axis=1)
    df = pd.concat([df.drop('marital-status', axis=1), pd.get_dummies(df['marital-status']).add_prefix('marital-status_')], axis=1)
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race).add_prefix('race_')], axis=1)
    df = pd.concat([df.drop('native-country', axis=1), pd.get_dummies(df['native-country']).add_prefix('native-country_')], axis=1)
    df = pd.concat([df.drop('relationship', axis=1), pd.get_dummies(df.relationship).add_prefix('relationship_')], axis=1)

    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    ```

3. **Visualize Correlations**:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18,12))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    ```

4. **Feature Selection**:
    ```python
    correlations = df.corr()['income'].abs()
    sorted_correlations = correlations.sort_values()
    num_of_cols_to_drop = int(0.8 * len(df.columns))
    cols_to_dorp = sorted_correlations.iloc[:num_of_cols_to_drop].index
    df_dropped = df.drop(cols_to_dorp, axis=1)

    plt.figure(figsize=(15,10))
    sns.heatmap(df_dropped.corr(), annot=True, cmap='coolwarm')
    ```

5. **Train the Classification Model**:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = df.drop('fnlwgt', axis=1)

    train_df, test_df = train_test_split(df, test_size=0.2)

    train_X = train_df.drop('income', axis=1)
    train_y = train_df['income']

    test_X = test_df.drop('income', axis=1)
    test_y = test_df['income']

    forest = RandomForestClassifier()
    forest.fit(train_X, train_y)
    forest.score(test_X, test_y)
    ```

6. **Feature Importance**:
    ```python
    importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
    importances = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}
    ```

7. **Hyperparameter Tuning**:
    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [50, 100, 250],
        'max_depth': [5, 10, 30, None],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose=10)
    grid_search.fit(train_X, train_y)

    forest = grid_search.best_estimator_
    forest.score(test_X, test_y)

    importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
    importances = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is from Kaggle: [Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset).
