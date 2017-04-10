import pandas as pd
import pickle
from time import time

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from best_parameters import GRID_SEARCH_PARAM


def get_data(pickle_file):
    # returns the unpickled dataframe
    # from the exploratory analysis
    print("Loading Pickled Data...")

    try:
        with open(pickle_file, 'rb') as p:
            dat = pickle.load(p)
            print("All good!")

            return dat
    except:
        print("Error. Is {} the correct path?".format(pickle_file))


def train_classifier(learner, X_train, y_train, X_valid, y_valid):
    # this function takes a classifier and data, optimizes
    # based on the parameters in GRID_SEARCH_PARAM, then
    # saves and returns the results
    results = dict()
    results['name'] = learner.__class__.__name__

    print("\nNow training {}...".format(results['name']))

    # Create a cross-validation strategy: Stratified K-fold
    cv_sets = StratifiedKFold(n_splits=5)

    # Make a scoring function: accuracy
    scorer = make_scorer(log_loss)

    # Time training
    start = time()

    # Grid search with cross validation
    grid = GridSearchCV(estimator=learner,
                        param_grid=GRID_SEARCH_PARAM[results['name']],
                        scoring='neg_log_loss',
                        cv=cv_sets)

    grid.fit(X_train, y_train)

    results['train_time'] = time() - start
    print("Training time: {:.2f} seconds".format(results['train_time']))

    # Predict training data
    train_pred = grid.predict(X_train)
    valid_pred = grid.predict(X_valid)

    # Output accuracies and record them in the results dictionary
    results['train_acc'] = accuracy_score(y_train, train_pred)
    print("Training accuracy: {:.3f}".format(results['train_acc']))
    results['valid_acc'] = accuracy_score(y_valid, valid_pred)
    print("Validation accuracy: {:.3f}".format(results['valid_acc']))
    results['best_params'] = grid.best_params_
    print("Best parameters from grid search: \n{}".format(results['best_params']))
    results['best_estimator'] = grid.best_estimator_

    return results


def format_submissions(learner, df_test):

    sample_submission_file = '../../submissions/sample_submission.csv'
    df_sample = pd.read_csv(sample_submission_file)

    df_submission = pd.DataFrame(columns=df_sample.columns)
    df_submission['id'] = df_test['id']
    X_test = df_test.drop(['id'], axis=1)

    preds = learner.predict_proba(X_test)

    class_mapping = dict()
    for i, col in enumerate(df_sample.columns[1:]):
        class_mapping[col] = i

    def classnum_to_name(num):
        for col, n in class_mapping.items():
            if n == num:
                return col


    for i,pred in enumerate(preds):
        for j,p in enumerate(pred):
            col = classnum_to_name(j)
            df_submission.set_value(i,col,p)

    new_sub_file = '../../submissions/' + 'Best' + learner.__class__.__name__ + '.csv'
    df_submission.to_csv(new_sub_file, index=False)
