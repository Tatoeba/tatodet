from tatodet.det import norm, upd_mdl, vec, get_tgms_frq, get_abs, new_dct
import numpy as np


def test_norm():
    r = np.zeros(5)
    r[:] = 100
    r[0] = 1000

    assert np.abs(norm(r).sum() - 1.0) <= 1e-4

snt = [
    'what is',
]

mk_dct = lambda s: get_abs(*get_tgms_frq(s), new_dct(), 'eng')

def test_update():
    d1 = mk_dct(['what'])
    d = upd_mdl(d1, mk_dct(['what']), 'eng')
    assert list(d['eng'].items())[0][1][0] == 4

