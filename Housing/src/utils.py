from collections import defaultdict
from numpy import log, abs, sqrt, exp, cov, ones, zeros, eye, array, nan
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

# conda install -c rasbt mlxtend
# Lets us do:
#   StackingRegressor(regressors=[ElasticNet(), RandomForestRegressor(), ...],
#                     meta_regressor=AvgReg())
from mlxtend.regressor import StackingRegressor

# conda install cvxopt
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# Painstakingly determined
fill_vals = {
  (2577, 'GarageType'): nan,
  (2127, 'GarageYrBlt'): 1963,
  (2127, 'GarageFinish'): "Unf",
  (2127, 'GarageQual'): "TA",
  (2127, 'GarageCond'): "TA",
  (2421, 'PoolQC'): "Gd",
  (2504, 'PoolQC'): "Gd",
  (2600, 'PoolQC'): "Gd",
  (1556, 'KitchenQual'): 'TA',
  (2611, 'MasVnrType'): 'BrkCmn',
  (1916, 'MSZoning'): 'RL',
  (2217, 'MSZoning'): 'RL',
  (2251, 'MSZoning'): 'RL',
  (2905, 'MSZoning'): 'RL',
  (27,   'BsmtExposure'): 'No',
  (332,  'BsmtFinType2'): 'Unf',
  (333,  'BsmtFinType2'): 'Unf',
  (580,  'BsmtCond'): 'TA',
  (725,  'BsmtCond'): 'TA',
  (888,  'BsmtExposure'): 'No',
  (948,  'BsmtExposure'): 'No',
  (949,  'BsmtExposure'): 'No',
  (1064, 'BsmtCond'): 'TA',
  (1488, 'BsmtExposure'): 'No',
  (2041, 'BsmtCond'): 'TA',
  (2186, 'BsmtCond'): 'TA',
  (2218, 'BsmtQual'): 'TA',
  (2219, 'BsmtQual'): 'TA',
  (2349, 'BsmtExposure'): 'No',
  (2525, 'BsmtCond'): 'TA',
  (1916, 'Utilities'): 'AllPub',
  (1946, 'Utilities'): 'AllPub',
  (2152, 'Exterior1st'): 'Wd Sdng',
  (2152, 'Exterior2nd'): 'MetalSd',
  (1380, 'Electrical'): 'SBrkr'
}

qualitatives = [('BsmtCond', 'HasBsmt'), ('GarageType',  'HasGarage'),
                ('PoolQC',   'HasPool'), ('FireplaceQu', 'HasFire'  ),
                ('Fence',   'HasFence'), ('Alley',       'HasAlley' ),
                ('SaleType','HasSaleT'), ('MiscFeature', 'HasMisc'  ),
                ('Functional','HasFunc')
                ]

def test_col_group(df, cols, hascol):
    coll = []
    for c in cols:
        coll.extend([x for x in df.columns if (c + '_') in x])
    assert(not df[coll][~df[hascol]].any().any())

def run_col_group_tests(df):
    test_col_group(df,
            ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'],
            'HasGarage')
    test_col_group(df,
            ['PoolArea', 'PoolQC'],
            'HasPool')
    test_col_group(df,
            ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
             'HasBsmt', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'],
            'HasBsmt')
    test_col_group(df, ['FireplaceQu', 'Fireplaces'], 'HasFire')
    test_col_group(df, ['Fence'], 'HasFence')
    test_col_group(df, ['Alley'], 'HasAlley')
    test_col_group(df, ['SaleType'], 'HasSaleT')
    test_col_group(df, ['MiscFeature'], 'HasMisc')
    test_col_group(df, ['Functional'], 'HasFunc')

def optimal_weighting(res, reg=.05):
    """
    Given a matrix of residuals, compute the minimum-variance weighting.
    The regularization term tends to distribute the weight more equally, and
    should roughly correspond to how much variance we are willing to allow in
    order to have the weights perfectly equal vs. fully concentrated.
    """
    nvar = res.shape[0]
    P = cvxopt.matrix(2 * cov(res) + 2 * reg * eye(nvar))
    q = cvxopt.matrix(zeros(nvar))
    G = cvxopt.matrix(-eye(nvar))
    h = cvxopt.matrix(zeros(nvar))
    A = cvxopt.matrix(ones((nvar,1))).T
    b = cvxopt.matrix([1.])
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    return array(sol['x']).reshape(-1)

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

def each_fold(fun, train, splits=5, shuffle=False):
    """
    Take a function on test and train data, and a dataset.
    Return the function applied to each fold of the dataset.
    """
    kf = KFold(n_splits=splits, shuffle=shuffle)
    return [fun(train.iloc[trn], train.iloc[tst]) for trn, tst in kf.split(train)]

def get_residual_fun(reg):
    """
    Takes a classifier and returns a function that can be passed to each_fold.
    The returned function will calculate the residual on each fold of the training set.
    """
    def res(train, test):
        reg.fit(train.drop(['SalePrice', 'Id'], axis=1), train['SalePrice'])
        pred = reg.predict(test.drop(['SalePrice', 'Id'], axis=1))
        return pd.DataFrame({'Id':test.Id, 'Residual':pred - test.SalePrice})
    return res


def get_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    #Log time
    train["SalePrice"] = log(train["SalePrice"])

    # Do numerical processing on these assholes
    ncols = [c for c, d in zip(train.columns, train.dtypes) if str(d) in ["float64", "int64"]]
    ncols.remove("Id")
    ncols.remove("SalePrice")
    for c in ncols:
        train[c].fillna(0, inplace=True)
        test[c].fillna(0, inplace=True)

    # Fill in the values we checked
    for ((i, col), val) in fill_vals.iteritems():
        i -= 1
        if i < train.shape[0]:
            train[col].iloc[i] = val
        else:
            test[col].iloc[i-train.shape[0]] = val

    cats = [c for c, d in zip(train.columns, train.dtypes) if str(d) == 'object']
    # Don't actually turn them into strings, as we're handling the nans more manually now
    # # Turn these fuckers into strings
    # for c in cats:
    #     train[c] = train[c].astype(str)
    #     test[c] = test[c].astype(str)

    # Special case for Masonry columns, as they already have a special 'None' value
    for x in (train, test):
        nomas = x.MasVnrType.isnull() & x.MasVnrArea.isnull()
        x.MasVnrType[nomas] = 'None'
        x.MasVnrArea[nomas] = 0.0

    # Set up our qualitative columns (whether house has a garage, etc.)
    for checkcol, name in qualitatives:
        train[name] = ~train[checkcol].isnull()
        test [name] = ~test [checkcol].isnull()
        
    # One hot these bitches (not in a sexist way)
    onehotvals = get_onehots(train, cats)
    set_onehots(train, onehotvals, drop=True)
    set_onehots(test, onehotvals, drop=True)

    # At this point, we should be passing our grouping tests
    run_col_group_tests(train)
    run_col_group_tests(test)
    
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

class WavgReg(BaseEstimator, RegressorMixin):

    def __init__(self, reg=0.05):
        self.w = None
        self.reg = reg

    def fit(self, X, y):
        res = X - y.values.reshape((-1, 1))
        self.w = optimal_weighting(res.T, self.reg)
        return self

    def predict(self, X):
        return ((X * self.w.reshape((1, X.shape[1]))).sum(axis=1) / self.w.sum()).reshape(-1)

