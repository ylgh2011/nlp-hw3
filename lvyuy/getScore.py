import os, optparse, sys

optparser = optparse.OptionParser()
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")

optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iteration", dest="iteration", default=5, type="int", help="The iteration number for the alignment learning.")

optparser.add_option("-w", "--whichScript", dest="whichScript", default='bi_is_joint.py', help="choose which script to run")
optparser.add_option("-o", "--outputFile", dest="outputFile", default='out.ignore', help="Name of the output file")

(opts, _) = optparser.parse_args()

cmd1  = 'python ' + opts.whichScript
cmd1 += ' -p ' + opts.fileprefix
cmd1 += ' -f ' + opts.french
cmd1 += ' -n ' + str(opts.num_sents)
cmd1 += ' -i ' + str(opts.iteration)
cmd1_noOut = cmd1
cmd1 += ' > ' + opts.outputFile

cmd2  = 'python score-alignments.py -i ' + opts.outputFile
cmd2_noOut = 'python score-alignments.py'

cmd = cmd1 + ' ; ' + cmd2

if opts.outputFile == 'noOut':
	cmd = cmd1_noOut + ' | ' + cmd2_noOut

print cmd
os.system(cmd);
