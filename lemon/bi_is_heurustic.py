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
optparser.add_option("-i", "--iteration", dest="iteration", default=5, type="int", help="The iteration number for the alignment learning.")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
sys.stderr.write("Training with IBM 2 method coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


# init_prob_t = 1.0 / float(len(voc_f.keys()))
init_prob_t = 1.0 / 30.0
init_prob_q = 1.0 / 30.0


def line_match(f, e, t_fe, q_fe, t_ef, q_ef, fe_count, e_count, jilm_count, ilm_count):
    m = len(f)
    l = len(e)

    for (i, f_i) in enumerate(f):
        norm_z = 0
        for (j, e_j) in enumerate(e):
            norm_z += (t_fe.get((f_i, e_j), init_prob_t)*q_fe.get((j,i,l,m), init_prob_q)*
                       t_ef.get((e_j, f_i), init_prob_t)*q_ef.get((i,j,m,l), init_prob_q))

        for (j, e_j) in enumerate(e):
            cnt = (t_fe.get((f_i, e_j), init_prob_t)*q_fe.get((j,i,l,m), init_prob_q)*
                   t_ef.get((e_j, f_i), init_prob_t)*q_ef.get((i,j,m,l), init_prob_q))/norm_z
            fe_count[f_i, e_j] += cnt
            e_count[e_j] += cnt
            jilm_count[j,i,l,m] += cnt
            ilm_count[i,l,m] += cnt


def update_dictionary(t, q, fe_count, e_count, jilm_count, ilm_count):
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
            if (swap):
                line_alg.append("{}-{}".format(bestj,i))
            else:
                line_alg.append("{}-{}".format(i,bestj))

    return line_alg


def align_compare(align_1, align_2):
    return int(align_1.split('-')[0]) - int(align_2.split('-')[0])


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
        t_fe = defaultdict(float)
        q_fe = defaultdict(float)

        t_ef = defaultdict(float)
        q_ef = defaultdict(float)

        sys.stderr.write("\nAsigning variable")
        update_dictionary(t_fe, q_fe, fe_count, e_count, jilm_fe_count, ilm_fe_count)
        update_dictionary(t_ef, q_ef, ef_count, f_count, jilm_ef_count, ilm_ef_count)

    sys.stderr.write("\nOutputing")
    for (f, e) in bitext:
        line_alg = list(set(alignment_line(f, e, t_fe, t_ef, q_fe, q_ef)).intersection
                   (set(alignment_line(e, f, t_ef, t_fe, q_ef, q_fe, True))))
        # heuristic matching
        sorted(line_alg, cmp=align_compare)
        lasti = -1
        pre_diff = 0
        for (i, align) in enumerate(line_alg):
            diff = int(align.split('-')[0]) - int(align.split('-')[1])
            if pre_diff == diff:
                for j in range(lasti + 1, i):
                    sys.stdout.write('{}-{} '.format(j, j-diff))
            lasti = i
            pre_diff = diff

        for align in line_alg:
            sys.stdout.write(align+' ')
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()
