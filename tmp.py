import pymongo

def get_client():
    client = pymongo.MongoClient('59.72.109.90',27017)
    return client

def query(client, corpora, feat_name, collection='ae_results'):
    cluster_db = client.cluster_db
    results = cluster_db[collection]
    query_results = results.find({'corpora':corpora, 'feat_name':feat_name})
    results = query_results[0]
    return results

def parse_results(results):
    acc_mean,acc_std = results['acc_mean'], results['acc_std']
    ari_mean,ari_std = results['ari_mean'], results['ari_std']
    nmi_mean,nmi_std = results['nmi_mean'], results['nmi_std']
    template = '${:.2f}\\pm{:.2f}\\%$'
    return ' | '.join([template.format(acc_mean*100, acc_std*100),
        template.format(nmi_mean*100, nmi_std*100),
        template.format(ari_mean*100, ari_std*100)])
