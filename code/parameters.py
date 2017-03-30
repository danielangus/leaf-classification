# -----Logging Parameters-----
LOG_PARAM = {
    'logging' : True,
    'logfile' : open('logs/feature_log.txt', 'w')
}

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
    'DecisionTreeClassifier': {'min_samples_split': [80, 120, 160]},
    'RandomForestClassifier': {'n_estimators': [100, 150, 200],
                               'min_samples_split': [80, 120, 160]},
    'AdaBoostClassifier': {'learning_rate': [0.7, 0.8, 0.9]},
    'LogisticRegression': {'C': [0.001, 0.01, 0.1],
                           'solver': ['newton-cg', 'lbfgs']},
    'SGDClassifier': {'loss': ['hinge', 'log'],
                      'penalty': ['none', 'l2', 'l1']}
}

