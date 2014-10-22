#!/usr/bin/env python
import optparse, sys, os, logging, copy
from collections import defaultdict
from math import log

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iteration", dest="iteration", default=5, type="int", help="The iteration number for the alignment learning.")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
sys.stderr.write("Training with IBM 2 method coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


# init_prob_t = 1.0 / float(len(voc_f.keys()))
init_prob_t = 1.0 / 30.0
init_prob_q = 1.0 / 30.0

gamma = 1.0


def init_t(t_fe, t_ef, bitext):
    C = defaultdict(float)
    nPairs = float(len(bitext))
    for f, e in bitext:
        for f_i in set(f):
            for e_j in set(e):
                C[f_i, e_j] += 1.0
        for e_j in set(e):
            C[e_j] += 1.0
        for f_i in set(f):
            C[f_i] += 1.0

    LLR = defaultdict(float)
    Z_e = defaultdict(float)
    Z_f = defaultdict(float)
    for f, e in bitext:
        for f_i in set(f):
            for e_j in set(e):
                LLR[f_i, e_j] = getLLR(f_i, e_j, C, nPairs)
                Z_f[f_i] += LLR[f_i, e_j]
                Z_e[e_j] += LLR[f_i, e_j]

    for f, e in bitext:
        for f_i in set(f):
            for e_j in set(e):
                t_fe[f_i, e_j] = (LLR[f_i, e_j] / Z_e[e_j]) if Z_e[e_j] != 0.0 else 0.0
                t_ef[e_j, f_i] = (LLR[f_i, e_j] / Z_f[f_i]) if Z_f[f_i] != 0.0 else 0.0


def getLLR(f, e, C, n):
    if C[f, e] * n < gamma * C[f] * C[e]:
        return 0.0
    f_e   = ( C[f, e] * log( C[f, e] * n / C[f] / C[e] ) ) if C[f, e] != 0.0 else 0.0
    nf_e  = ( (C[e] - C[f, e]) * log( (C[e] - C[f, e]) * n / (n - C[f]) / C[e] ) ) if (C[e] - C[f, e]) != 0.0 else 0.0
    f_ne  = ( (C[f] - C[f, e]) * log( (C[f] - C[f, e]) * n / C[f] / (n - C[e]) ) ) if (C[f] - C[f, e]) != 0.0 else 0.0
    nf_ne = ( (n - C[f] - C[e] + C[f, e]) * log( (n - C[f] - C[e] + C[f, e]) * n / (n - C[f]) / (n - C[e]) ) ) if (n - C[f] - C[e] + C[f, e]) != 0.0 else 0.0
    return f_e + nf_e + f_ne + nf_ne


def line_match(f, e, t_fe, q_fe, t_ef, q_ef, fe_count, e_count, jilm_count, ilm_count):
    m = len(f)
    l = len(e)

    for (i, f_i) in enumerate(f):
        norm_z_t = 0.0
        norm_z_q = 0.0
        for (j, e_j) in enumerate(e):
            norm_z_t += t_fe.get((f_i, e_j), init_prob_t) * t_ef.get((e_j, f_i), init_prob_t)
            norm_z_q += q_fe.get((j,i,l,m), init_prob_q) * q_ef.get((i,j,m,l), init_prob_q)

        for (j, e_j) in enumerate(e):
            cnt = t_fe.get((f_i, e_j), init_prob_t) * t_ef.get((e_j, f_i), init_prob_t) / norm_z_t if norm_z_t != 0.0 else 0.0
            fe_count[f_i, e_j] += cnt
            e_count[e_j] += cnt
            cnt = q_fe.get((j,i,l,m), init_prob_q) * q_ef.get((i,j,m,l), init_prob_q) / norm_z_q if norm_z_q != 0.0 else 0.0
            jilm_count[j,i,l,m] += cnt
            ilm_count[i,l,m] += cnt


def update_dictionary(t, q, fe_count, e_count, jilm_count, ilm_count):
    # update t dictionary
    for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
        t[f_i, e_j] = fe_count[f_i, e_j]/e_count[e_j] if e_count[e_j] != 0 else 0.0
        if k % 50000 == 0:
            sys.stderr.write(".")

    # update q dictionary
    for (k, (j, i, l, m)) in enumerate(jilm_count.keys()):
        q[j, i, l, m] = jilm_count[j,i,l,m]/ilm_count[i,l,m] if ilm_count[i,l,m] != 0 else 0.0
        if k % 5000 == 0:
            sys.stderr.write(".")


def alignment_line(f, e, t_fe, t_ef, q_fe, q_ef, swap=False):
    line_alg = []
    for (i, f_i) in enumerate(f):
        bestp = 0
        bestj = 0 
        m = len(f)
        l = len(e)
        for (j, e_j) in enumerate(e):
            mat = t_fe[f_i, e_j]*t_ef[e_j, f_i]*q_fe[j, i, l, m]*q_ef[i, j, m, l]
            if  mat > bestp:
                bestp = mat 
                bestj = j

        if bestp != 0:
            if swap:
                line_alg.append("{}-{}".format(bestj,i))
            else:
                line_alg.append("{}-{}".format(i,bestj))

    return line_alg


def main():
    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    # Initialization step
    # voc_f = defaultdict()
    # for (n, (f, e)) in enumerate(bitext):
    #     for f_i in set(f):
    #         voc_f[f_i] = 1


    t_fe = defaultdict(float)
    q_fe = defaultdict(float)

    t_ef = defaultdict(float)
    q_ef = defaultdict(float)

    init_t(t_fe, t_ef, bitext)

    for iter_cnt in range(opts.iteration):
        sys.stderr.write("\nTraining")
        # inherit last iteration

        # init count 
        fe_count = defaultdict(float)
        e_count = defaultdict(float)
        jilm_fe_count = defaultdict(float)
        ilm_fe_count = defaultdict(float)

        ef_count = defaultdict(float)
        f_count = defaultdict(float)
        jilm_ef_count = defaultdict(float)
        ilm_ef_count = defaultdict(float)

        for (n, (f, e)) in enumerate(bitext):
            # match e to f
            line_match(f, e, t_fe, q_fe, t_ef, q_ef, fe_count, e_count, jilm_fe_count, ilm_fe_count)
            # match f to e
            line_match(e, f, t_ef, q_ef, t_fe, q_fe, ef_count, f_count, jilm_ef_count, ilm_ef_count)

            # process indicator        
            if n % 500 == 0:
                sys.stderr.write(".")

        # clean up t_prev and q_prev
        # t_fe = defaultdict(float)
        # q_fe = defaultdict(float)

        # t_ef = defaultdict(float)
        # q_ef = defaultdict(float)

        sys.stderr.write("\nAssigning variable")
        update_dictionary(t_fe, q_fe, fe_count, e_count, jilm_fe_count, ilm_fe_count)
        update_dictionary(t_ef, q_ef, ef_count, f_count, jilm_ef_count, ilm_ef_count)

    sys.stderr.write("\nOutputing")
    for (f, e) in bitext:
        line_alg = (set(alignment_line(f, e, t_fe, t_ef, q_fe, q_ef)).intersection
                   (set(alignment_line(e, f, t_ef, t_fe, q_ef, q_fe, True))))
        for i in line_alg:
            sys.stdout.write(i+' ')
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()
