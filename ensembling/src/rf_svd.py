import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

def run_training(fold):
    df = pd.read_csv("ensembling/input/train_folds.csv")
    df.review = df.review.apply(str)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    # print(xtrain[0, :])

    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(xtrain_svd, ytrain)
    pred = clf.predict_proba(xvalid_svd)[:, -1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "rf_svd_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "rf_svd_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)
    
    fin_valid_df = pd.concat(dfs)
    fin_valid_df.to_csv("ensembling/model_preds/rf_svd.csv", index=False)