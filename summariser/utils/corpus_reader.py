import sys
import os.path as path
import codecs

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import re, os
from resources import DOC_SEQUENCE_PATH

class CorpusReader(object):
    def __init__(self, base_path, parse_type=None):
        self.base_path = base_path
        self.parse_type=parse_type

    def getDocs(self,path):
        ele = path.split('/')
        dataset = ele[-3]
        topic = ele[-2]
        docs = None

        ff = open(DOC_SEQUENCE_PATH,'r')
        for line in ff.readlines():
            if '{};{}'.format(dataset,topic) in line:
                docs = line.split(':')[1].split(';')
                for ii in range(len(docs)):
                    docs[ii] = docs[ii].strip()
                return docs

        ff.close()
        if docs is None:
            print('INVALID PATH: {}'.format(path))
            exit(11)

    def load_processed(self, path, summary_len=None):
        data = []

        if summary_len is not None:
            docs = os.listdir(path)
            summaries = [model for model in docs if re.search("M\.%s\." % (summary_len), model)]
            docs = summaries
        else:
            docs = self.getDocs(path)

        for doc_name in docs:
            filename = "%s/%s" % (path, doc_name)
            with codecs.open(filename, 'r', 'utf-8') as fp:
                text = fp.read().splitlines()
            data.append((filename, text))
        return data

    def get_data(self, corpus_name, summary_len=100):
        """
        generator function that returns a iterable tuple which contains

        :rtype: tuple consisting of topic, contained documents, and contained summaries
        :param corpus_name: 
        :param summary_len:
        """
        corpus_base_dir = path.join(self.base_path, corpus_name)

        docs_directory_name = "docs"
        models_directory_name = "summaries"
        if self.parse_type == "parse":
            docs_directory_name = "docs.parsed"
            models_directory_name = "summaries.parsed"

        dir_listing = os.listdir(corpus_base_dir)
        for ctopic in sorted(dir_listing):
            docs_path = path.join(corpus_base_dir, ctopic, docs_directory_name)
            summary_path = path.join(corpus_base_dir, ctopic, models_directory_name)

            docs = self.load_processed(docs_path)
            summaries = self.load_processed(summary_path, summary_len)
            yield ctopic, docs, summaries
