from det import build_mdl, save_mdl, langs


#TODO add cli options, orm queries
if __name__ == '__main__':
    dct_lng, snt_lng, mx_lng = build_mdl()
    save_mdl(dct_lng)

