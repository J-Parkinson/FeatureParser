from loadData import *
from evaluate import *
from myFeatureBasedParser import *

def runClassifier():
    train = loadData("train")
    train_features = extract_features(train)
    #Now we need to train based on these features
    X_train = train_features.drop(['token', 'label', 'bio_only', 'upos'], axis=1)
    #We drop upos since we already have the POS as an integer.
    y_train = train_features['bio_only']
    
    train_evaluate = wnut_evaluate(train_features)

runClassifier()