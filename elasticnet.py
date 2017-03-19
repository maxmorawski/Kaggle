#!/usr/bin/env python

from sklearn.linear_model import ElasticNet
import utils as u

best_ratio = .073
best_alpha = 0.00058

if __name__ == '__main__':
    train, vals = u.get_train_data()
    clf = ElasticNet(alpha=best_alpha, l1_ratio=best_ratio)
    cv = u.run_cross_val(clf, train)
    print("Elastic Net: alpha={}, l1_ratio={}".format(best_alpha, best_ratio))
    print("Expected performance: {}".format(cv))
    u.build_submission(clf, train=train, onehotvals=vals, fname='data/submission2.csv')
