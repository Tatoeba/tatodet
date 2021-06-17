import numpy as np
import pymc3 as pm
import theano as th
import matplotlib.pyplot as plt
import dill as pkl
import re

from sklearn.feature_extraction.text import CountVectorizer as cvec
from collections import defaultdict as ddict
from operator import itemgetter as get
from langs import langs


def plt_hist(pt, rng=[0, 1]):
    bins = np.linspace(rng[0], rng[1], 200)
    histogram, bins = np.histogram(pt, bins=bins, density=True)

    bin_centers = 0.5*(bins[1:] + bins[:-1])


    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, histogram, label="Histogram of samples")

def new_dct(flr=0.5, mx=1000):
    return ddict(lambda: ddict(lambda : np.array([flr, mx-flr])))

def vec(snt):
    v = cvec(ngram_range=(3, 3), analyzer='char')
    tgm = v.fit_transform(snt)
    nm = v.get_feature_names()
    return v, tgm, nm

def get_tgms_frq(snt):
    v, tgm, nm = vec(snt)
    aps = tgm.sum(axis=0)
    return nm, aps

def get_tgms_cnt(snt):
    v, tgm, nm = vec(snt)
    tgm[tgm > 1] = 1
    aps = tgm.sum(axis=0)
    return nm, aps

def get_abs(nm, aps, dct, lang, mx=1000):
    aps = np.array(aps.astype(float))
    for tg, ap in zip(nm, aps.T):
        a = ap[0] + 1
        dct[lang][tg] = np.array([a, mx-a])
    return dct


def mdl_beta(ab, n=1):
    a, b = ab[:, 0], ab[:, 1]
    bt = pm.Beta.dist(alpha=a+0.5, beta=b)
    probs = bt.random(size=n)
    probs = probs.reshape((n, -1))
    return probs

def mdl_pbeta(ab, n=1):
    a, b = ab[:, 0], ab[:, 1]
    ap, bp = pm.Poisson.dist(a), pm.Poisson.dist(b)
    a_, b_ = ap.random(size=n), bp.random(size=n)
    bt = pm.Beta.dist(alpha=a_+0.5, beta=b_)
    probs = bt.random(size=1)
    probs = probs.reshape((n, -1))
    return probs

def norm(cnt):
    probs = cnt/cnt.sum()
    return probs
    

upr = norm(np.ones(len(langs)))

def predict(snt, dct, prior=upr, n=1, mdl=mdl_pbeta):
    v, tgm, nm = vec([snt])

    lng_pb = []
    for idx, lang in enumerate(langs):
        ab = np.vstack(get(*nm)(dct[lang]))
        probs = mdl(ab, n=n)
        lng_pb.append(np.prod(probs, axis=1))

    lng_pb = np.vstack(lng_pb).T
    pr = prior

    pb = pr * lng_pb
    preds = pb.argmax(axis=1)
    vs, cs = np.unique(preds, return_counts=True)
    pred = vs[cs.argmax()]

    return langs[pred], (pred, preds, (vs, cs), lng_pb, pr, pb)
    
def predict_lg(snt, dct, prior=upr, n=1, mdl=mdl_pbeta):
    v, tgm, nm = vec([snt])

    lng_pb = []
    for idx, lang in enumerate(langs):
        ab = np.vstack(get(*nm)(dct[lang]))
        probs = mdl(ab, n=n)
        lng_pb.append(np.log(probs).sum(axis=1))

    lng_pb = np.vstack(lng_pb).T
    pr = np.log(prior)
    
    pb = pr + lng_pb
    preds = pb.argmax(axis=1)
    vs, cs = np.unique(preds, return_counts=True)
    pred = vs[cs.argmax()]

    return langs[pred], (pred, preds, (vs, cs), lng_pb, pr, pb)

def load_csv(fl='data/testset.csv'): 
    with open(fl, 'r') as f: 
        snt = [i.split('\t') for i in f.read().split('\n')] 
 
        X = [] 
        Y = [] 
        for s in snt: 
            if len(s) == 3 and len(s[2]) >= 4: 
                X.append(s[2]) 
                Y.append(langs.index(s[1])) 
 
    return X, Y 

def load_lng(lng, fl='./data/%s_1000.csv'):
    fl = fl % (lng)
    
    with open(fl) as f:
        snt = f.read()
        snt = re.sub('^.*\t', '', snt, flags=re.MULTILINE)
        snt = snt.split('\n')

    return snt



def build_mdl(langs=langs, fl='./data/%s_1000.csv'):
    mx_lng = {}
    snt_lng = {}
    dct_lng = new_dct()
    for lg in langs:
        snt = load_lng(lg, fl=fl)
        snt_lng[lg] = snt
        mx_lng[lg] = len(snt)

        tg_nm, aps = get_tgms_frq(snt)
        dct_lng = get_abs(tg_nm, aps, dct_lng, lg, mx=mx_lng[lg])

    return dct_lng, snt_lng, mx_lng


def save_mdl(dct, fl='model.pkl'):
    with open(fl, 'wb') as f:
        pkl.dump(dct, f)

def load_mdl(fl='model.pkl'):
    with open(fl, 'rb') as f:
        dct = pkl.load(f)

    return dct

def upd_mdl(dct, ndct, lng, mx=1000): 
    for k, v in ndct[lng].items(): 
        o_ap = dct[lng][k] 
        n_ap = v 
        ap = n_ap[0] + o_ap[0] 
        dct[lng][k] = np.array([ap, mx-ap]) 
 
    return dct 

try:
    dct_lng = load_mdl()
    print('loading existing model')
except e:
    print('building and saving new model')
    dct_lng, snt_lng, mx_lng = build_mdl()
    save_mdl(dct_lng)
