from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from langs import langs
from det import load_csv, mdl_beta, mdl_pbeta, predict, predict_lg, vec
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

n_cpu = 2

v = lambda n1, n2: lambda x: vec(x, ngm=(n1, n2))
pred = lambda n, mdl, dc, vc: lambda s: predict(s, dc, n=n, mdl=mdl, vct=vc)[1][0]
pl_pred = lambda x, prd: Parallel(n_jobs=n_cpu, require='sharedmem')(delayed(prd)(s) for s in x)
cm_plt = lambda y, yh: ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y, yh), display_labels=langs).plot()

def score_rpt(y, yh, nmx='1'):
    acc = accuracy_score(y, yh)
    print(acc)
    cm = confusion_matrix(y, yh)
    print(cm)
    cm_plt(y, yh)
    plt.savefig('yhat_%s.png' % nmx)

if __name__ == '__main__':
    x, y = load_csv('data/testset.csv')
    dct_t = load_mdl(fl='model_t.pkl')
    dct_bt = load_mdl(fl='model_bt.pkl')
    dct_ubt = load_mdl(fl='model_ubt.pkl')

    yh1 = pl_pred(x, pred(n=30, mdl=mdl_pbeta, dc=dct_t, vc=v(3,3)))
    score_rpt(y, yh1, nmx=1)
    yh2 = pl_pred(x, pred(n=30, mdl=mdl_beta, dc=dct_t, vc=v(3,3)))
    score_rpt(y, yh2, nmx=2)
    yh3 = pl_pred(x, pred(n=30, mdl=mdl_beta, dc=dct_bt, vc=v(2,3)))
    score_rpt(y, yh3, nmx=3)
    yh4 = pl_pred(x, pred(n=30, mdl=mdl_beta, dc=dct_ubt, vc=v(1,3)))
    score_rpt(y, yh4, nmx=4)
