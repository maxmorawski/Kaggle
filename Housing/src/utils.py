from numpy import log, abs, sqrt, exp
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score

INT_FILLNA_COLS = [
        'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
TRAIN_IGNORE = ['GarageYrBlt', 'Id', 'SalePrice']
TEST_IGNORE = ['GarageYrBlt', 'Id']
TARGET_COL = 'SalePrice'

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

def get_train_data(int_fillna_cols=INT_FILLNA_COLS, one_hot=True):
    """
    Read the training data from disk, identify the categorical variables and
    values to one-hot encode, and return the encoded data and the dictionary
    of one-hot values.
    """
    train = pd.read_csv('../data/train.csv')
    train[TARGET_COL] = log(train[TARGET_COL])
    if(int_fillna_cols):
        for col in int_fillna_cols:
            train[col].fillna(0, inplace=True)
    if(not one_hot):
        return train
    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    onehotvals = get_onehots(train, cats)
    set_onehots(train, onehotvals, drop=True)
    return train, onehotvals

def get_test_data(onehotvals, int_fillna_cols=INT_FILLNA_COLS):
    """
    Read the test data from disk, and one-hot encode the previously identified
    columns and values.
    """
    test = pd.read_csv('data/test.csv')
    for col in int_fillna_cols:
        test[col].fillna(0, inplace=True)
    set_onehots(test, onehotvals, drop=True)
    return test

def run_cross_val(clf, train, cv=5):
    scores = cross_val_score(clf, train.drop(TRAIN_IGNORE, axis=1), train[TARGET_COL],
            cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    return sqrt(abs(scores)).mean()

def build_submission(clf,
        fit=True, onehotvals=None, train=None, test=None, fname='data/submission.csv'):
    if (fit and train is None) or onehotvals is None:
        train, onehotvals = get_train_data()
    if fit:
        clf.fit(train.drop(TRAIN_IGNORE, axis=1), train[TARGET_COL])
    if test is None:
        test = get_test_data(onehotvals)
    pred = clf.predict(test.drop(TEST_IGNORE, axis=1))
    sub = pd.DataFrame({'Id': test.Id, 'SalePrice': exp(pred)})
    sub.to_csv(fname, index=False)

