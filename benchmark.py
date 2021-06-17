from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from langs import langs
from det import load_csv, mdl_beta, mdl_pbeta, dct_lng, predict, predict_lg
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

n_cpu = 2

pred = lambda n, mdl: lambda s: predict(s, dct_lng, n=n, mdl=mdl)[1][0]
pl_pred = lambda x, prd: Parallel(n_jobs=n_cpu, require='sharedmem')(delayed(prd)(s) for s in x)
cm_plt = lambda y, yh: ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y, yh), display_labels=langs).plot()

def score_rpt(y, yh, nmx='1'):
    acc = accuracy_score(y, yh)
    print(acc)
    cm = confusion_matrix(y, yh)
    print(cm)
    cm_plt(y, yh)
    plt.savefig('yhat_%s.png' % nmx)

x, y = load_csv('data/testset.csv')


yh1 = pl_pred(x, pred(n=30, mdl=mdl_pbeta))
yh2 = pl_pred(x, pred(n=30, mdl=mdl_beta))
score_rpt(y, yh1, nmx=1)
score_rpt(y, yh2, nmx=2)
