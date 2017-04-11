
import pickle
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelBinarizer

def get_data(pickle_file):
    # returns the unpickled dataframe
    # from the exploratory analysis
    try:
        with open(pickle_file, 'rb') as p:
            data = pickle.load(p)
            return data
    except:
        print("Error. Is {} the correct path?".format(pickle_file))


# Load data
train_pickle_file = '../../data/pickles/train_data.pkl'
test_pickle_file = '../../data/pickles/test_data.pkl'

df_train = get_data(train_pickle_file)
df_test = get_data(test_pickle_file)

# Make training data numpy matrices
y = df_train['species_num'].as_matrix()
X = df_train.drop(['id', 'species_num'], axis=1).as_matrix()

# One-hot encode training labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Make testing data numpy matrix
X_test = df_test.drop(['id'], axis=1).as_matrix()

# Create Keras model
model = Sequential()

model.add(Dense(units=512, input_dim=192))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=99))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          epochs=40,
          batch_size=32,
          verbose=1,
          shuffle=True,
          validation_split=0.2)

# Predict test data
output_preds = model.predict(X_test)

# Format submission and output to csv
sample_submission_file = '../../submissions/sample_submission.csv'
df_sample = pd.read_csv(sample_submission_file)
ids = df_sample['id']
df_sample = df_sample.drop(['id'], axis=1)
outputs = pd.DataFrame(output_preds, columns=df_sample.columns)
outputs['id'] = ids
outputs.to_csv('../../submissions/nn.csv', index=False)