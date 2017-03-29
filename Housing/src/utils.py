from collections import defaultdict
from numpy import log, abs, sqrt, exp
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

# conda install -c rasbt mlxtend
# Lets us do:
#   StackingRegressor(regressors=[ElasticNet(), RandomForestRegressor(), ...],
#                     meta_regressor=AvgReg())
from mlxtend.regressor import StackingRegressor

def cross_val(clf, train, cv=5):
    scores = cross_val_score(clf, train.drop(["SalePrice", "Id"], axis=1), train["SalePrice"],
                             cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    return sqrt(abs(scores)).mean()

def get_onehots(df, cols):
    """
    Creat a dictionary of lists, where each list corresponds to the unique
    non-null values in a particular column of the dataframe.
    """
    vals = defaultdict(list)
    for c in cols:
        for v in df[c].dropna().unique():
            vals[c].append(v)
    return dict(vals)

def set_onehots(df, vals, drop=True):
    """
    Take a dictionary as created by get_onehots and create one-hot encoded
    columns for each value of each column of interest.
    """
    for c in vals.keys():
        for v in vals[c]:
            df[c + '_' + str(v)] = df[c].apply(lambda x: 1 if x == v else 0)
        if drop:
            df.drop(c, inplace=True, axis=1)

def get_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    #Log time
    train["SalePrice"] = log(train["SalePrice"])

    # Sadbad column
    train = train.drop('GarageYrBlt', axis=1)
    test  =  test.drop('GarageYrBlt', axis=1)

    # Do numerical processing on these assholes
    ncols = [c for c, d in zip(train.columns, train.dtypes) if str(d) in ["float64", "int64"]]
    ncols.remove("Id")
    ncols.remove("SalePrice")
    for c in ncols:
        train[c].fillna(0, inplace=True)
        test[c].fillna(0, inplace=True)
    
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    # Turn these fuckers into strings
    for c in cats:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        
    # One hot these bitches (not in a sexist way)
    onehotvals = get_onehots(train, cats)
    set_onehots(train, onehotvals, drop=True)
    set_onehots(test, onehotvals, drop=True)
    
    maxs = [train[c].max() for c in ncols]
    mins = [train[c].min() for c in ncols]
    for c, mx, mn in zip(ncols, maxs, mins):
        train[c] = (train[c] - mn) / (mx - mn)
        test[c] = (test[c] - mn) / (mx - mn)

    return train, test

def save_submission(test, pred, fname):
    res = pd.DataFrame({'Id': test.Id, 'SalePrice': exp(pred)})
    res.to_csv(fname, index=False)

def build_submission(clf, train, test, fname=None):
    clf.fit(train.drop(['SalePrice', 'Id'], axis=1), train.SalePrice)
    pred = clf.predict(test.drop('Id', axis=1))
    if fname is not None:
        save_submission(test, pred, fname)
    return pred

# Regressor that just takes the average of its inputs.
class AvgReg(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return X.mean(axis=1).reshape(-1)
