from featureParser import *
from loadData import *
from evaluate import *
from trainClassifier import *

def runClassifier():
    train = loadData("train")
    train_features = extract_features(train)
    trained_classifier = trainModel(train_features)
    train_evaluate = wnut_evaluate(trained_classifier)

runClassifier()