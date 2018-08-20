import sys, os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
PROJECT_PATH = path.dirname(path.dirname(path.abspath(__file__)))

from summariser.data_processer.corpus_cleaner import CorpusCleaner

def main():

    corpus_name = 'DUC2004' # DUC2001, DUC2002, DUC2004
    parse_type = 'parse'
    language = 'english'
    data_path = path.join(PROJECT_PATH, "../data")

    if parse_type !=None and language==None:
        raise AttributeError('Please specify language')

    corpus = CorpusCleaner(data_path, corpus_name, parse_type, language)
    if corpus_name[:3] == 'DUC' or corpus_name[:3] == 'TAC':
        corpus.cleanDuc_data(parse_type)
    if corpus_name[:3] == 'DBS':
        corpus.cleanDBS_data(parse_type)
    if corpus_name == 'WikiAIPHES':
        corpus.cleanWiki_data(parse_type)
    else:
        pass

if __name__ == '__main__':
    main()
