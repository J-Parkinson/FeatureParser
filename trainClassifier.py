from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def getXTrain(txt):
    return txt.drop(['token', 'label', 'bio_only', 'upos'], axis=1)

def getYTrain(txt):
    return txt['bio_only']

def splitEvenlyIntoBI_O_Segments(txt):
    is_inside = txt['bio_only'] != 'O'
    is_outside = txt['bio_only'] == 'O'
    bi = txt[is_inside]
    outside = txt[is_outside]
    outside = outside.sample(n=3000)  # approx the sum of B and I labels in train

    # recombine
    train = pd.concat([bi, outside])
    print('Down-sampled data:')
    train.bio_only.value_counts()

def trainModel(txt):
    txtSampled = splitEvenlyIntoBI_O_Segments(txt)
    X_train = getXTrain(txtSampled)
    Y_train = getYTrain(txtSampled)

    logreg = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(
        X_train, Y_train)

    return logreg

def testModel(txt, logreg):
    X_dev = getXTrain(txt)
    Y_dev = getYTrain(txt)
    preds = logreg.predict(X_dev)

    (unique, counts) = np.unique(preds, return_counts=True)
    print('Predicted label, Count of labels')
    print(np.asarray((unique, counts)).T)

