#!/usr/bin/env python
import optparse, sys, os, logging, copy
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
sys.stderr.write("Training with IBM 2 method coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


def main():
    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    # Initialization step
    # voc_f = defaultdict()
    # for (n, (f, e)) in enumerate(bitext):
    #     for f_i in set(f):
    #         voc_f[f_i] = 1

    # init_prob_t = 1.0 / float(len(voc_f.keys()))
    init_prob_t = 1.0 / 30.0
    init_prob_q = 1.0 / 30.0


    t = defaultdict(float)
    q = defaultdict(float)

    for iter_cnt in range(5):
        sys.stderr.write("\nTraining")
        # inherit last iteration

        # init count 
        fe_count = defaultdict(float)
        e_count = defaultdict(float)
        jilm_count = defaultdict(float)
        ilm_count = defaultdict(float)
        for (n, (f, en)) in enumerate(bitext):
            m = len(f)
            e = copy.deepcopy(en)
            e.insert(0,'NULL')
            l = len(e)
            for (i, f_i) in enumerate(f):
                norm_z = 0
                for (j, e_j) in enumerate(e):
                    norm_z += t.get((f_i, e_j), init_prob_t)*q.get((j,i,l,m), init_prob_q)

                for (j, e_j) in enumerate(e):
                    cnt = t.get((f_i, e_j), init_prob_t)*q.get((j,i,l,m), init_prob_q)/norm_z
                    fe_count[f_i, e_j] += cnt
                    e_count[e_j] += cnt
                    jilm_count[j,i,l,m] += cnt
                    ilm_count[i,l,m] += cnt

            # process indicator        
            if n % 500 == 0:
                sys.stderr.write(".")

        # clean up t_prev and q_prev
        t = defaultdict(float)
        q = defaultdict(float)
        sys.stderr.write("\nAsigning variable")
        # update t dictionary
        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            t[f_i, e_j] = fe_count[f_i, e_j]/e_count[e_j]
            if k % 50000 == 0:
                sys.stderr.write(".")

        # update q dictionary
        for (k, (j, i, l, m)) in enumerate(jilm_count.keys()):
            q[j, i, l, m] = jilm_count[j,i,l,m]/ilm_count[i,l,m]
            if k % 5000 == 0:
                sys.stderr.write(".")





    sys.stderr.write("\nOutputing")
    for (f, en) in bitext:
        e = copy.deepcopy(en)
        e.insert(0,'NULL')
        for (i, f_i) in enumerate(f):
            bestp = 0
            bestj = 0 
            for (j, e_j) in enumerate(e):
                if t[f_i, e_j]*q[j,i,l,m] > bestp:
                    bestp = t[f_i, e_j]*q[j,i,l,m]
                    bestj = j
            if e[bestj] != 'NULL':
                # subtract bestj with 1 since with addition of NULL word, the position of english word is shifted right
                bestj = bestj - 1
                sys.stdout.write("%i-%i " % (i,bestj))
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()