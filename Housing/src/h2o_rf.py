#!/usr/bin/env python
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import utils
import pandas as pd
from numpy import exp

train = utils.get_train_data(one_hot=False)
test = utils.get_test_data()

train_frm = h2o.H2OFrame(train)
test_frm = h2o.H2OFrame(test)

clf = H2ORandomForestEstimator()
cols = train_frm.col_names
cols.remove('SalePrice')
cols.remove('Id')
clf.train(x=cols, y='SalePrice', training_frame=train_frm)
res = clf.predict(test_frm).as_data_frame()

out = pd.DataFrame({'Id': test.Id, 'SalePrice': exp(res.predict)})
out.to_csv('../data/submission3.csv', index=False)
