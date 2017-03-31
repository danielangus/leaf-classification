# -----Neural Network Parameters-----
NN_PARAM = {
    'layer1_neurons': 256,
    'layer2_neurons': 512,
    'layer3_neurons': 256,
    'output_classes': 99,
    'dropout': True,
    'keep_prob': 0.7,
    'l2_reg': True,
    'beta': 5e-3,
    'batch_size': 32,
    'num_steps': 10001,
    'learning_rate': 0.001,
    'decay_steps': 500,
    'decay_rate': 0.95,
    'staircase': True,
    'verbose' : False
}

# -----Grid Search Parameters-----
GRID_SEARCH_PARAM = {
    'DecisionTreeClassifier': {'min_samples_split': [2,3,4,5]},
    'RandomForestClassifier': {'n_estimators': [100, 150, 200, 250],
                               'min_samples_split': [2, 3, 4, 5]},
    'AdaBoostClassifier': {'learning_rate': [0.01, 0.05, 0.1]},
    'LogisticRegression': {'C': [15.0, 20.0, 25.0],
                           'solver': ['liblinear', 'newton-cg', 'lbfgs']},
    'SGDClassifier': {'loss': ['hinge', 'log'],
                      'penalty': ['none', 'l2', 'l1']}
}