import xgboost

import glob
import numpy as np
import pandas as pd

from sklearn import metrics

from functools import partial
from scipy.optimize import fmin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -1.0 * auc_score

    def fit(self, X, y):
        partial_loss = partial(self._auc, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=True)
    
    def predict(self, X):
        x_coef = X * self.coef_
        predicitons = np.sum(x_coef, axis=1)
        return predicitons
    
def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xvalid = scaler.transform(xvalid)

    clf = xgboost.XGBClassifier()
    clf.fit(xtrain, train_df.sentiment.values)
    preds = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")
    valid_df.loc[:, "xgb_pred"] = preds
    return valid_df

if __name__ == "__main__":
    files = glob.glob("ensembling/model_preds/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
    targets = df.sentiment.values
    pred_cols = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]
    
    dfs = []
    for j in range(5):
        temp_df = run_training(df, j)
        dfs.append(temp_df)
    
    fin_valid_df = pd.concat(dfs)
    print(metrics.roc_auc_score(fin_valid_df.sentiment.values, fin_valid_df.xgb_pred.values))