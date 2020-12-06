import glob
import json
import csv

result_filename = 'result.csv'
column_names = ['corpora', 'feat_name', 'n_topics', 'best_acc', 'best_nmi', 'best_ari']
with open(result_filename,'w') as fw:
    writer = csv.writer(fw)
    writer.writerow(column_names)
    # fname = 'lda_results.txt'
    for fname in glob.glob('*_results.txt'):
        with open(fname) as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    d = json.loads(l)
                    row = [d[c] for c in column_names]
                    writer.writerow(row)
                except:
                    continue


