from bottle import route, run, request
from det import predict, predict_lg, mdl_pbeta, mdl_beta, load_mdl
from benchmark import v
import numpy as np

pred_dct = {
    0: predict,
    1: predict_lg
}

mdl_dct = {
    0: mdl_pbeta,
    1: mdl_beta
}

dct = load_mdl()

@route('/v1/det', method='GET')
def det():
    d = {}
    q = dict(request.query)

    mdli = int(q.get('method', 0))
    assert mdli == 0 or mdli == 1, 'incorrect method'
    mdl = mdl_dct[mdli]

    log = int(q.get('log', 0))
    assert log == 0 or log == 1, 'incorrect log'
    pred = pred_dct[log]

    ver = int(q.get('verbose', 0))
    assert ver == 0 or ver == 1, 'incorrect verbose value'

    tr = int(q.get('trials', 10))
    assert tr >=1 and tr <= 100, 'trials must be between 1 and 100'

    n1, n2 = q.get('ngram', '1,3').split(',')
    ngm = (int(n1), int(n2))

    sent = q.get('sent', '')
    assert len(sent) > 3, 'sent must be greater than 3 chars'


    res = pred(sent, dct, mdl=mdl, n=tr, vct=v(*ngm))
    d['lang'] = res[0]
    if ver:
        d['verbose'] = str(res)

    return d


run(host='localhost', port=8080, debug=True)
