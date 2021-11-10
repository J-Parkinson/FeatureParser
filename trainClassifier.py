from sklearn.linear_model import LogisticRegression
import numpy as np

numbersToLetters = "BIO"

def getXTrain(txt):
    return txt.drop(['token', 'label', 'bio_only', 'upos'], axis=1)

def getYTrain(txt):
    return txt['bio_only']

def trainModels(txt):
    X_train = getXTrain(txt)
    Y_train = getYTrain(txt)

    logreg = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg', class_weight="balanced").fit(
        X_train, Y_train)

    logReg2 = trainModelNGram(txt, logreg)

    return logreg, logReg2

def trainModelNGram(txt, logreg):
    X_train = getXTrain(txt)
    Y_train = getYTrain(txt)
    #Probably should use dev set here instead?
    preds = fixPreds(logreg.predict(X_train))
    X_train["preds-1"] = np.roll(preds, 1)
    X_train["preds-2"] = np.roll(preds, 2)
    X_train["preds+1"] = np.roll(preds, -1)
    X_train["preds+2"] = np.roll(preds, -1)
    logreg = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg',
                                class_weight="balanced").fit(
        X_train, Y_train)

    return logreg


def fixPreds(preds):
    for i in range(1, len(preds)):
        if preds[i] in [0, 1]:
            if preds[i-1] in [0, 1]:
                preds[i] = 1
            else:
                preds[i] = 0
    return preds

def saveResults(txt, saveLoc):
    txtFilter = txt[["token", "upos", "prediction"]]
    txtFilter.to_csv(saveLoc, sep="\t")
    txt.to_csv("testFull.txt", sep="\t")

def testModel(txt, logregs, save=True, saveLoc="test.txt"):
    X_dev = getXTrain(txt)
    Y_dev = getYTrain(txt)
    preds_1gram = fixPreds(logregs[0].predict(X_dev))
    X_dev["preds-1"] = np.roll(preds_1gram, 1)
    X_dev["preds-2"] = np.roll(preds_1gram, 2)
    X_dev["preds+1"] = np.roll(preds_1gram, -1)
    X_dev["preds+2"] = np.roll(preds_1gram, -2)
    print(X_dev.head(50))
    preds = fixPreds(logregs[1].predict(X_dev))

    (unique, counts) = np.unique(preds, return_counts=True)
    print('Predicted label, Count of labels')
    print(np.asarray((unique, counts)).T)

    (unique, counts) = np.unique(Y_dev, return_counts=True)
    print('Actual label, Count of labels')
    print(np.asarray((unique, counts)).T)

    txt['prediction'] = [numbersToLetters[pred] for pred in preds]
    txt["bio_only"] = [numbersToLetters[bio] for bio in txt["bio_only"]]

    if save:
        saveResults(txt, saveLoc)

    return txt

