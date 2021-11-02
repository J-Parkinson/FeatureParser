from sklearn.linear_model import LogisticRegression
import numpy as np

def getXTrain(txt):
    return txt.drop(['token', 'label', 'bio_only', 'upos'], axis=1)

def getYTrain(txt):
    return txt['bio_only']

def trainModel(txt):
    X_train = getXTrain(txt)
    Y_train = getYTrain(txt)

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

