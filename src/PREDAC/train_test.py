from sklearn.naive_bayes import BernoulliNB

def train(dataset):
    model = BernoulliNB()
    model.fit(dataset.x, dataset.y)
    return model

def test(dataset, model):
    y_pred = model.predict_proba(dataset.x)[:,1]
    return y_pred

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_val_idx = train_idx + val_idx
    optied_model = train(dataset.subset(train_val_idx))
    return test(dataset.subset(test_idx), optied_model)