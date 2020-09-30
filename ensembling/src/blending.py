import glob
import numpy as np
import pandas as pd

from sklearn import metrics

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
    # print(df.head(10))

    pred_cols = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]

    for col in pred_cols:
        auc = metrics.roc_auc_score(df.sentiment.values, df[col].values)
        print(f"{col}, overall_auc={auc}")

    print("1. average")
    avg_pred = np.mean(df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values, axis=1)
    print(metrics.roc_auc_score(targets, avg_pred))

    print("2. weighted average")
    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    rf_svd_pred = df.rf_svd_pred.values

    #weights should be optimize
    #also we can use model voting for predictions
    avg_pred = (3 * lr_pred + lr_cnt_pred + rf_svd_pred) / 5
    print(metrics.roc_auc_score(targets, avg_pred))

    print("3. rank average")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (lr_pred + lr_cnt_pred + rf_svd_pred) / 3
    print(metrics.roc_auc_score(targets, avg_pred))

    print("4. weighted rank average")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values
    avg_pred = (3 * lr_pred + lr_cnt_pred + rf_svd_pred) / 5
    print(metrics.roc_auc_score(targets, avg_pred))

    


