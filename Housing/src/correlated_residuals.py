from utils import *
import utils
from numpy import cov, ones, zeros, eye, array

if __name__ == '__main__':
    # Our best regressors so far
    clf1 = ElasticNet(alpha=.00062, l1_ratio=.825)
    clf2 = SVR(C=9, gamma=.0025)
    clf3 = GradientBoostingRegressor(learning_rate=.1, n_estimators=200)
    clf4 = RandomForestRegressor(200)
 
    train, test = utils.get_data()

    # Compute the residuals of each regressor using a k=5 fold over the training data
    clfs = [clf1, clf2, clf3, clf4]
    ress = [pd.concat( utils.each_fold( utils.get_residual_fun(c), train) ) for c in clfs]
    resmat = array([r.Residual for r in ress])

    # Use the covariance of the residuals to compute the optimal weighting of the regressors
    wav = utils.WavgReg(.008)
    stack = StackingRegressor(regressors=clfs[:3], meta_regressor=wav)

    utils.build_submission(stack, train, test, '../data/submission12.csv')
