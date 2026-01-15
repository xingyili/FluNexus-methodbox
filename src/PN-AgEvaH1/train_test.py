from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from model import *
import time
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse


def train(dataset):
    model = PNAgEvaH1()
    model.fit(dataset.x, dataset.y)
    return model

def test(dataset, model):
    y_pred = model(dataset.x)
    return y_pred

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_val_idx = train_idx + val_idx
    train_dataset = dataset.subset(train_val_idx)
    test_dataset = dataset.subset(test_idx)
    optied_model = train(train_dataset)
    return test(test_dataset, optied_model)