from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

def getXTrain(txt):
    return txt.drop(['token', 'label', 'bio_only', 'upos'], axis=1)

def getYTrain(txt):
    return txt['bio_only']

def trainModel(txt):
    X_train = getXTrain(txt)
    Y_train = getYTrain(txt)

    dectree = DecisionTreeClassifier(random_state=0).fit(
        X_train, Y_train)

    return dectree

