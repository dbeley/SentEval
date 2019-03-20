from __future__ import absolute_import, division, unicode_literals
import sys
import os
import torch
import logging
import time
# sent2vec
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

# sent2vec
FASTTEXT_EXEC_PATH = os.path.abspath("../../sent2vec/fasttext")

# BASE_SNLP_PATH = "/~/Documents/Données/stanford-postagger-full-2018-10-16/"
# SNLP_TAGGER_JAR = os.path.join(BASE_SNLP_PATH, "stanford-postagger-3.9.2.jar")
SNLP_TAGGER_JAR = os.path.abspath("../../Données/stanford-postagger-full-2018-10-16/stanford-postagger-3.9.2.jar")

MODEL_TORONTOBOOKS_UNIGRAMS = os.path.abspath("../../Données/torontobooks_unigrams.bin")
# MODEL_TORONTOBOOKS_BIGRAMS = os.path.abspath("./sent2vec_wiki_bigrams")


def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence


def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token


def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]


def get_embeddings_for_preprocessed_sentences(sentences, model_path, fasttext_exec_path):
    """Arguments:
        - sentences: a list of preprocessed sentences
        - model_path: a path to the sent2vec .bin model
        - fasttext_exec_path: a path to the fasttext executable
    """
    timestamp = str(time.time())
    test_path = os.path.abspath('./'+timestamp+'_fasttext.test.txt')
    embeddings_path = os.path.abspath('./'+timestamp+'_fasttext.embeddings.txt')
    dump_text_to_disk(test_path, sentences)
    call(fasttext_exec_path + ' print-sentence-vectors ' + model_path + ' < ' + test_path + ' > ' + embeddings_path, shell=True)
    embeddings = read_embeddings(embeddings_path)
    os.remove(test_path)
    os.remove(embeddings_path)
    assert(len(sentences) == len(embeddings))
    return np.array(embeddings)


def read_embeddings(embeddings_path):
    """Arguments:
        - embeddings_path: path to the embeddings
    """
    with open(embeddings_path, 'r') as in_stream:
        embeddings = []
        for line in in_stream:
            line = '['+line.replace(' ',',')+']'
            embeddings.append(eval(line))
        return embeddings
    return []


def dump_text_to_disk(file_path, X, Y=None):
    """Arguments:
        - file_path: where to dump the data
        - X: list of sentences to dump
        - Y: labels, if any
    """
    with open(file_path, 'w') as out_stream:
        if Y is not None:
            for x, y in zip(X, Y):
                out_stream.write('__label__'+str(y)+' '+x+' \n')
        else:
            for x in X:
                out_stream.write(x+' \n')


def get_sentence_embeddings(sentences, ngram='bigrams', model='concat_wiki_twitter'):
    """ Returns a numpy matrix of embeddings for one of the published models. It
    handles tokenization and can be given raw sentences.
    Arguments:
        - ngram: 'unigrams' or 'bigrams'
        - model: 'wiki', 'twitter', or 'concat_wiki_twitter'
        - sentences: a list of raw sentences ['Once upon a time', 'This is another sentence.', ...]
    """
    toronto_embeddings = None
    tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
    # tknzr = CoreNLPParser(SNLP_TAGGER_JAR, encoding='utf-8')
    s = ' <delimiter> '.join(sentences) #just a trick to make things faster
    tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
    tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
    assert(len(tokenized_sentences_SNLP) == len(sentences))
    toronto_embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, MODEL_TORONTOBOOKS_UNIGRAMS, FASTTEXT_EXEC_PATH)
    return toronto_embeddings
    sys.exit(-1)


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
