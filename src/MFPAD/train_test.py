import xgboost as xgb
def train(dataset=None):
    model = xgb.XGBRegressor(
        max_depth=13,
        booster='gbtree',
        gamma='0.1',
        learning_rate=0.1,
        n_estimators=200,
        objective='reg:squarederror',
        seed=250
    )
    model.fit(dataset.x, dataset.y)
    return model

def test(dataset, model):
    y_pred = model.predict(dataset.x)
    return y_pred

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    train_val_idx = train_idx + val_idx
    optied_model = train(dataset.subset(train_val_idx))
    return test(dataset.subset(test_idx), optied_model)