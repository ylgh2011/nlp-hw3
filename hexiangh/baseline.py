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
sys.stderr.write("Training with Baseline method coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


def main():
    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    # Initialization step
    voc_f = defaultdict(int)
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            voc_f[f_i] = 1

    init_prob = 1.0/len(voc_f.keys())
    t_prev = defaultdict(int)
    t_cur = defaultdict(int)

    iter_cnt = 0
    for iter_cnt in range(5):
        sys.stderr.write("\nTraining")
        # inherit last iteration
        t_prev = copy.deepcopy(t_cur)
        t_cur = defaultdict(int)

        # init count 
        fe_count = defaultdict(int)
        e_count = defaultdict(int)
        for (n, (f, e)) in enumerate(bitext):
            for f_i in set(f):
                norm_z = 0
                for e_j in set(e):
                    norm_z += t_prev.get((f_i, e_j), init_prob)

                for e_j in set(e):
                    cnt = t_prev.get((f_i, e_j), init_prob)/norm_z
                    fe_count[f_i, e_j] += cnt
                    e_count[e_j] += cnt

            # process indicator        
            if n % 500 == 0:
                sys.stderr.write(".")

        sys.stderr.write("\nAsigning variable")
        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            t_cur[f_i, e_j] = fe_count[f_i, e_j]/e_count[e_j]
            if k % 5000 == 0:
                sys.stderr.write(".")



    sys.stderr.write("\nOutputing")
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            bestp = 0
            bestj = 0 
            for (j, e_j) in enumerate(e):
                if t_cur[f_i, e_j] > bestp:
                    bestp = t_cur[f_i, e_j]
                    bestj = j

            sys.stdout.write("%i-%i " % (i,bestj))
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()