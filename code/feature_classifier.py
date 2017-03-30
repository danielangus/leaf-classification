from nn_functions import run_tf_nn
from feature_functions import get_data, printer, run_base_classifiers

# -----CLASSIFIERS-----
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

def main():
    train_pickle_file = '../data/pickles/train_data.pkl'
    test_pickle_file = '../data/pickles/test_data.pkl'

    df_train = get_data(train_pickle_file)
    df_test = get_data(test_pickle_file)

    y = df_train['species_num']
    X = df_train.drop(['id', 'species_num'], axis=1)

    X_train, y_train, X_test, y_test = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=0)

    results = run_tf_nn(X_train, y_train, X_test, y_test)

    printer(results, pretty=True)



if __name__=='__main__':
    main()