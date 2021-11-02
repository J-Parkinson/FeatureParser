from featureParser import *
from loadData import *
from evaluate import *
from trainClassifier import *

def runClassifier():
    train = loadData("train")
    dev = loadData("dev")
    train_features = extract_features(train, preparetxt=True)
    trained_classifier = trainModel(train_features)
    testModel(extract_features(dev), trained_classifier)
    #train_evaluate = wnut_evaluate(trained_classifier)

runClassifier()