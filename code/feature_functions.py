# -----MAIN LIBRARIES-----
import numpy as np
# -----HELPER LIBRARIES-----
import pickle
from pprint import pprint
from time import time
# -----SCORING AND MODEL SELECTION-----
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# -----PARAMETERS-----
from feature_classifier import LOG_PARAM
from feature_classifier import GRID_SEARCH_PARAM

# -----Processing Functions-----
def get_data(pickle_file):
    # returns the unpickled dataframe
    # from the exploratory analysis
    printer("Loading Pickled Data...")

    try:
        with open(pickle_file, 'rb') as p:
            dat = pickle.load(p)
            printer("All good!")

            return dat
    except:
        printer("Error. Is {} the correct path?".format(pickle_file))

def printer(args, pretty=False):
    # printer function to handle cases with logging and pretty printing
    logging = LOG_PARAM['logging']
    logfile = open('log.txt', 'w')
    if logging and not pretty:
        print('\n', args, end="", file=logfile)
        print(args)
    elif logging and pretty:
        pprint(args, logfile)
        pprint(args)
    elif not logging and pretty:
        pprint(args)
    elif not logging and not pretty:
        print(args)

# -----Training Functions-----
def train_classifier(learner, X_train, y_train, X_test, y_test):
    # this function takes a classifier and data, optimizes
    # based on the parameters in GRID_SEARCH_PARAM, then
    # saves and returns the results
    results = dict()
    results['name'] = learner.__class__.__name__

    # I was getting an error with grid search
    # Github discussion suggests it is because of train_test_split
    # So I added this step:
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=np.int32)

    printer("\nNow training {}...".format(results['name']))

    # Create a cross-validation strategy: Stratified K-fold
    cv_sets = StratifiedKFold(n_splits=5)

    # Make a scoring function: accuracy
    scorer = make_scorer(accuracy_score)

    # Time training
    start = time()

    # Grid search with cross validation
    grid = GridSearchCV(estimator=learner,
                        param_grid=GRID_SEARCH_PARAM[results['name']],
                        scoring=scorer,
                        cv=cv_sets)

    grid.fit(X_train, y_train)

    results['train_time'] = time() - start
    printer("Training time: {:.2f} seconds".format(results['train_time']))

    # Predict training data
    train_pred = grid.predict(X_train)
    test_pred = grid.predict(X_test)

    # Output accuracies and record them in the results dictionary
    results['train_acc'] = accuracy_score(y_train, train_pred)
    printer("Training accuracy: {:.3f}".format(results['train_acc']))
    results['test_acc'] = accuracy_score(y_test, test_pred)
    printer("Testing accuracy: {:.3f}".format(results['test_acc']))
    results['best_params'] = grid.best_params_
    printer("Best parameters from grid search: \n{}".format(results['best_params']))

    # Examine the confusion matrix for the classifier
    printer("Confusion matrix: ")
    printer(confusion_matrix(y_test, test_pred))

    return results


def run_base_classifiers(classifier_list, X_train, y_train, X_test, y_test):
    # this function handles the various classifiers, adding them to
    # a list before training each individually
    printer("Running base classifiers...")

    results = list()
    for classifier in classifier_list:
        results.append(train_classifier(classifier, X_train, y_train, X_test, y_test))

    return results
