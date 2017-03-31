
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from feature_functions import get_data, train_classifier, format_submissions

def get_classifier_list():

    classifier_list = list()
    classifier_list.append(DecisionTreeClassifier(random_state=0))
    classifier_list.append(RandomForestClassifier(random_state=1))
    classifier_list.append(AdaBoostClassifier(random_state=2))
    classifier_list.append(LogisticRegression(random_state=3))
    classifier_list.append(SGDClassifier(random_state=4))

    return classifier_list


def main():
    train_pickle_file = '../data/pickles/train_data.pkl'
    test_pickle_file = '../data/pickles/test_data.pkl'

    df_train = get_data(train_pickle_file)
    df_test = get_data(test_pickle_file)

    y = df_train['species_num']
    X = df_train.drop(['id', 'species_num'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=0)

    classifier_list = get_classifier_list()
    results = list()

    for classifier in classifier_list:
        results.append(train_classifier(classifier, X_train, y_train, X_valid, y_valid))

    for result in results:
        best_estimator = result['best_estimator']
        format_submissions(best_estimator, df_test)



if __name__=='__main__':
    main()