#!/usr/bin/env python3
import os, time, csv, numpy as np, tensorflow as tf, deepchem as dc
from sklearn.metrics import accuracy_score
np.random.seed(456); tf1=tf.compat.v1; tf1.disable_eager_execution()
def load_splits():
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w
    return (train_X,train_y[:,0],train_w[:,0]),(valid_X,valid_y[:,0],valid_w[:,0]),(test_X,test_y[:,0],test_w[:,0])
def build_graph(d,n_hidden=50,n_layers=1,lr=1e-3):
    g=tf.Graph()
    with g.as_default():
        x=tf1.placeholder(tf.float32,(None,d)); y=tf1.placeholder(tf.float32,(None,)); w=tf1.placeholder(tf.float32,(None,)); kp=tf1.placeholder(tf.float32)
        last=x; dim=d
        for L in range(n_layers):
            W=tf1.Variable(tf.random.normal((dim,n_hidden))); b=tf1.Variable(tf.random.normal((n_hidden,)))
            last=tf.nn.dropout(tf.nn.relu(tf.matmul(last,W)+b), rate=1.0-kp); dim=n_hidden
        W=tf1.Variable(tf.random.normal((dim,1))); b=tf1.Variable(tf.random.normal((1,)))
        ylog=tf.matmul(last,W)+b; yprob=tf.sigmoid(ylog); ypred=tf.round(yprob)
        ent=tf.nn.sigmoid_cross_entropy_with_logits(logits=ylog, labels=tf.expand_dims(y,1))
        loss=tf.reduce_sum(tf.expand_dims(w,1)*ent)
        train=tf1.train.AdamOptimizer(lr).minimize(loss); merged=tf1.summary.merge([tf1.summary.scalar('loss',loss)])
        init=tf1.global_variables_initializer()
    return g,dict(x=x,y=y,w=w,kp=kp,ypred=ypred,loss=loss,train=train,merged=merged,init=init)
def iter_mb(X,y,w,bs):
    N=X.shape[0]; idx=np.arange(N); np.random.shuffle(idx)
    for s in range(0,N,bs):
        sl=idx[s:s+bs]; yield X[sl],y[sl],w[sl]
def train_once(p):
    (trX,trY,trW),(vX,vY,vW),_ = load_splits(); d=trX.shape[1]
    g,t=build_graph(d,p['n_hidden'],p['n_layers'],p['lr'])
    with tf1.Session(graph=g) as sess:
        sess.run(t['init']); step=0
        for e in range(p['epochs']):
            for bX,bY,bW in iter_mb(trX,trY,trW,p['bs']):
                sess.run([t['train']],feed_dict={t['x']:bX,t['y']:bY,t['w']:bW,t['kp']:p['dropout']}); step+=1
        vpred=sess.run(t['ypred'],feed_dict={t['x']:vX,t['kp']:1.0}); return float(accuracy_score(vY,vpred, sample_weight=vW))
def main():
    grid={'n_hidden':[50,128],'n_layers':[1,2],'lr':[0.001,0.0005],'epochs':[30,45],'bs':[64,100],'dropout':[0.5,0.7]}
    repeats=3; results=[]; out='tuning_results.csv'
    keys=list(grid.keys())
    import itertools
    for values in itertools.product(*[grid[k] for k in keys]):
        p=dict(zip(keys,values)); print('\nConfig:',p); scores=[train_once(p) for _ in range(repeats)]
        mean=float(np.mean(scores)); std=float(np.std(scores)); row=dict(p); row.update({'valid_acc_mean':mean,'valid_acc_std':std,'repeats':repeats}); results.append(row)
        write_header=not os.path.exists(out)
        with open(out,'a',newline='') as f: w=csv.DictWriter(f,fieldnames=list(row.keys())); 
        # Write header then row
        with open(out,'a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        print(f'Mean Valid Acc: {mean:.4f} ± {std:.4f}')
    best=max(results,key=lambda r:r['valid_acc_mean']); print('\nBEST:',best)
if __name__=='__main__': main()
