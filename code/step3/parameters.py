# -----Grid Search Parameters-----
GRID_SEARCH_PARAM = {
    'LogisticRegression': {'C': [50.0, 75.0, 100.0, 125.0, 150.0, 200.0],
                           'solver': ['newton-cg', 'lbfgs'],
			   'fit_intercept': [True, False],
			   'max_iter': [50, 100, 150, 200],
			   'tol': [1e-2, 1e-3, 1e-4, 1e-5] }
}
