#!/usr/bin/env python3
import numpy as np; np.random.seed(456)
import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w
train_y, valid_y, test_y = train_y[:,0], valid_y[:,0], test_y[:,0]
train_w, valid_w, test_w = train_w[:,0], valid_w[:,0], test_w[:,0]
m = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=456)
print('About to fit model on train set.'); m.fit(train_X, train_y)
for name,(X,y,w) in {'train':(train_X,train_y,train_w),'valid':(valid_X,valid_y,valid_w),'test':(test_X,test_y,test_w)}.items():
    pred=m.predict(X); ws=accuracy_score(y,pred,sample_weight=w); print(f'Weighted {name} Classification Accuracy: {ws:.6f}')