from __future__ import absolute_import, division, unicode_literals import sys import os
import torch
import logging
import time
import numpy as np
import re
from subprocess import call
from nltk.tokenize.stanford import StanfordTokenizer
# from nltk.parse.corenlp import CoreNLPParser

STARTTIME = time.time()

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '../../Données/crawl-300d-2M.vec'  # or crawl-300d-2M.vec for V2
MODEL_PATH = '../../Données/infersent1.pkl'

sys.path.insert(0, PATH_SENTEVAL)
import senteval

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

def prepare(params, samples):
    # params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)
    return


def batcher(params, batch):
    # sent2vec
    # print(batch)
    batch = [' '.join(sent) for sent in batch]
    # print(batch)
    embeddings = get_sentence_embeddings(batch, ngram='unigrams')
    # print(embeddings)
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    print("Not implemented yet.")
    exit()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['Length', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
    runtime = time.time() - STARTTIME
    print("Runtime : %.2f seconds" % runtime)
