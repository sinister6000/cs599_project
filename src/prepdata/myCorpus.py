# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
__metaclass__ = type

import codecs
import os
import types

import twokenize
from gensim import corpora, utils
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


class MyCorpus(object):
    """
    Class that creates a gensim-compatible corpus (i.e., an iterable that yields Bag-of-Words versions of documents,
    one at a time). All linguistic pre-processing needs to happen before transforming a document into a BOW.
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            foundOneArg = True
            theOnlyArg = args[0]
        else:
            foundOneArg = False
            theOnlyArg = None

        if foundOneArg and isinstance(theOnlyArg, types.StringType):
            self.initializeFromFile(theOnlyArg)
        else:
            self.initializeFromArgs(*args)


    def initializeFromFile(self, fname):
        super(MyCorpus, self).__init__(fname)
        self.tokenizer = fname[16:-10]


    def initializeFromArgs(self, *args):
        self.top_dir = args[0]
        self.tokenizer = args[1]
        self.dictionary = corpora.Dictionary(self.iter_documents())

        # remove tokens that appear in only one document
        self.dictionary.filter_extremes(no_below=2, no_above=1.0, keep_n=None)
        self.dictionary.compactify()


    def __iter__(self):
        for tokens in self.iter_documents():
            yield self.dictionary.doc2bow(tokens)


    def iter_documents(self):
        """
        Helper function for MyCorpus.__iter__()
        Iterate over all documents in top_directory, yielding a document (=list of utf8 tokens) at a time.
        """
        ven_ids = sorted(os.listdir('../../data/ven'))
        make_ven_index()

        for doc_file in ven_ids:
            with codecs.open('../../data/ven/' + doc_file, 'r', encoding='utf-8') as fin:
                doc = fin.read()
                tokens = tokenize(doc, self.tokenizer)
                yield tokens


# END class MyCorpus


def make_ven_index():
    """
    Create a file that links venue.ID with the offset in MmCorpus.docbyoffset.

    :return: None
    :rtype: None
    """
    with codecs.open('../../data/ven_id2i.txt', 'w', encoding='utf-8') as ven_id2i:
        for i, doc in enumerate(sorted(os.listdir('../../data/ven'))):
            ven_id2i.write('{0}\t{1}\n'.format(doc[:-4], i))


def tokenize(s, tokenizer):
    """
    Tokenizes a string. Returns a different list of tokens depending on which tokenizer is used.

    :param s: string to be tokenized
    :type s: str
    :param tokenizer: identifies tokenizer to use
    :type tokenizer: str
    :return: list of tokens
    :rtype: []
    """
    tokens = (twokenize.tokenize(s)
              if tokenizer is 'twokenize'
              else (utils.tokenize(s, lower=True)
                    if tokenizer is 'gensim'
                    else (TweetTokenizer(preserve_case=False)).tokenize(s)))

    # list of symbols that can end sentences. twokenize has found these to not be attached to another token.
    # (safe to remove)
    punct = r'.,!!!!????!:;'

    # NLTK english stopwords
    stoplist = stopwords.words('english')

    result = [tok.lower() for tok in tokens if tok not in punct]
    result = [tok for tok in result if tok not in stoplist]
    return result


if __name__ == '__main__':
    pass
