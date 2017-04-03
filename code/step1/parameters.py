# -----Grid Search Parameters-----
GRID_SEARCH_PARAM = {
    'DecisionTreeClassifier': {'min_samples_split': [2,3,4,5]},
    'RandomForestClassifier': {'n_estimators': [100, 150, 200, 250],
                               'min_samples_split': [2, 3, 4, 5]},
    'AdaBoostClassifier': {'learning_rate': [0.01, 0.05, 0.1]},
    'LogisticRegression': {'C': [10.0, 12.0, 15.0, 20.0, 25.0],
                           'solver': ['liblinear', 'newton-cg', 'lbfgs']},
    'SGDClassifier': {'loss': ['hinge', 'log'],
                      'penalty': ['none', 'l2', 'l1']},
    'SVC': {'C': [40.0, 50.0, 60.0],
            'probability': [True]}
}