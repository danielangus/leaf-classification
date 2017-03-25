import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_file = '../data/pickles/train_data.pkl'
test_file = '../data/pickles/test_data.pkl'
df_train_dataset =  pd.read_pickle(train_file)
df_test_dataset = pd.read_pickle(test_file)

submission_file = '../submissions/sample_submission.csv'
df_sample = pd.read_csv(submission_file)

class_mapping=dict()
for i,col in enumerate(df_sample.columns[1:]):
    class_mapping[col] = i

y_train = df_train_dataset['species_num']
id_train = df_train_dataset['id']
X_train = df_train_dataset.drop(['id','species_num'], axis=1)

id_test = df_test_dataset['id']
X_test = df_test_dataset.drop(['id'], axis=1)

def train_predict(learner, X_train, y_train):
    print('learner: ', learner.__class__.__name__)

    learner.fit(X_train, y_train)
    preds = learner.predict(X_train)
    print('accuracy: ', accuracy_score(preds, y_train))

def classnum_to_name(num):
    for col,n in class_mapping.items():
        if n==num:
            return col

def test_predict(learner, X_test):
    preds = learner.predict_proba(X_test)

    df_submission = pd.DataFrame(columns=df_sample.columns)

    df_submission['id'] = id_test

    for i,pred in enumerate(preds):
        for j,p in enumerate(pred):
            col = classnum_to_name(j)
            df_submission.set_value(i,col,p)

    new_sub_file = '../submissions/' + learner.__class__.__name__ + '.csv'
    df_submission.to_csv(new_sub_file, index=False)

clf1 = DecisionTreeClassifier()
#clf2 = LogisticRegression()

print("---TRAINING---")
train_predict(clf1, X_train, y_train)
#train_predict(clf2, X_train, y_train)

print("---TESTING---")
submission = test_predict(clf1, X_test)
