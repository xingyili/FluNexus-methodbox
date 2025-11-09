from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

def train(dataset):
    model = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=1860, max_features=0.393686389369039),
        n_estimators=230,
        learning_rate=1.39248292746222,
    )
    model.fit(dataset.x, dataset.y)
    return model

def test(dataset, model):
    y_pred = model.predict(dataset.x)
    return y_pred

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_val_idx = train_idx + val_idx
    train_dataset = dataset.subset(train_val_idx)
    test_dataset = dataset.subset(test_idx)


    ohe_at_name = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    ohe_sr_name = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    at_name_encoded = ohe_at_name.fit_transform( [[x] for x in train_dataset.at_name])
    sr_name_encoded = ohe_sr_name.fit_transform( [[x] for x in train_dataset.sr_name])
    train_dataset.x = sparse.hstack((sparse.csr_matrix(train_dataset.x), at_name_encoded, sr_name_encoded)).tocsr()
    test_at_name_encoded = ohe_at_name.transform( [[x] for x in test_dataset.at_name])
    test_sr_name_encoded = ohe_sr_name.transform( [[x] for x in test_dataset.sr_name])
    test_dataset.x = sparse.hstack((sparse.csr_matrix(test_dataset.x), test_at_name_encoded, test_sr_name_encoded)).tocsr()

    optied_model = train(train_dataset)
    return test(test_dataset, optied_model)