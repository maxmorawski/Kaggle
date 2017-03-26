#!/usr/bin/env python

from numpy import exp
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import utils
import encode as enc

best_ratio = .73
best_alpha = 0.00058

if __name__ == '__main__':
    train = utils.get_train_data([], False)
    test = utils.get_test_data(None, [])
    ncols = [c for c, d in zip(train.columns, train.dtypes) if str(d) in ['float64', 'int64']]
    ncols.remove('SalePrice')
    ncols.remove('GarageYrBlt')
    ncols.remove('Id')
    ids = test.Id
    train, test = enc.fix_categorical(
            train.drop(utils.TEST_IGNORE, axis=1),
            test.drop(utils.TEST_IGNORE, axis=1), option='one_hot')
    enc.fix_numeric(train, test, ncols, scaling='uniform')

    clf3 = GradientBoostingRegressor()
    cv = utils.run_cross_val(clf3, train)
    print("GradientBoosting")
    print("Expected performance: {}".format(cv))
    print("")

    clf1 = ElasticNet(alpha=best_alpha, l1_ratio=best_ratio)
    cv = utils.run_cross_val(clf1, train)
    print("Elastic Net: alpha={}, l1_ratio={}".format(best_alpha, best_ratio))
    print("Expected performance: {}".format(cv))
    print("")

    clf2 = RandomForestRegressor(100)
    cv = utils.run_cross_val(clf2, train)
    print("RandomForest: n_estimators={}".format(100))
    print("Expected performance: {}".format(cv))
    print("")

    clf1.fit(train.drop('SalePrice', axis=1), train.SalePrice)
    clf2.fit(train.drop('SalePrice', axis=1), train.SalePrice)
    clf3.fit(train.drop('SalePrice', axis=1), train.SalePrice)

    p1 = clf1.predict(test)
    p2 = clf2.predict(test)
    p3 = clf3.predict(test)

    pred = (p1 + p2 + p3) / 3.0
    pd.DataFrame({'Id': ids, 'SalePrice': exp(pred)}).to_csv('../data/submission4.csv', index=False)
    pd.DataFrame({'Id': ids, 'SalePrice': exp(p3)}).to_csv('../data/submission5.csv', index=False)
