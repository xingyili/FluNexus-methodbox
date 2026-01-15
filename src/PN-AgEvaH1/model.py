from sklearn.ensemble import RandomForestClassifier
import numpy as np
class PNAgEvaH1():
    def __init__(self):
        self.model = RandomForestClassifier()

    def __call__(self, x):
        return self.forward(x)
    
    def fit(self, x, y):
        X_features = x[:, 1:]
        self.model.fit(X_features, y)

    def forward(self, x):
        x = np.array(x)

        n_samples = x.shape[0]
        y_pred = np.zeros(n_samples)

        hard_rule_indices = np.where(x[:, 0] == 1)[0]
        model_indices = np.where(x[:, 0] != 1)[0]

        if len(hard_rule_indices) > 0:
            y_pred[hard_rule_indices] = 1
            
        if len(model_indices) > 0:
            X_features = x[model_indices, 1:]
            preds = self.model.predict(X_features)
            y_pred[model_indices] = preds
            
        return y_pred