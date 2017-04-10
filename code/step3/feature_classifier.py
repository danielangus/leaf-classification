
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from feature_functions import get_data, train_classifier, format_submissions


def main():
    # Get data pickled in exploratory analysis
    train_pickle_file = '../../data/pickles/train_data.pkl'
    test_pickle_file = '../../data/pickles/test_data.pkl'

    df_train = get_data(train_pickle_file)
    df_test = get_data(test_pickle_file)

    # Make training and validation sets
    y = df_train['species_num']
    X = df_train.drop(['id', 'species_num'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=0)

    # Train classifiers on the data and get the best estimators
    clf = LogisticRegression(random_state=1)

    results = train_classifier(clf, X_train, y_train, X_valid, y_valid)

    # Run the best estimator on the test data and format it as a submission
    best_estimator = results['best_estimator']
    format_submissions(best_estimator, df_test)


if __name__=='__main__':
    main()