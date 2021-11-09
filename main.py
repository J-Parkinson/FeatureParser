from featureParser import *
from loadData import *
from evaluate import *
from trainClassifier import *

def runClassifier():
    train = loadData("train")
    dev = loadData("dev")
    test = loadData("test")

    #run with test

    test["upos"] = test["label"]
    test["bio_only"] = "B"

    train_features = extract_features(train)
    trained_classifiers = trainModels(train_features)
    txt_pred = testModel(extract_features(dev), trained_classifiers)

    #run with test
    #txt_pred = testModel(extract_features(test), trained_classifiers)

    wnut_evaluate(txt_pred)

runClassifier()