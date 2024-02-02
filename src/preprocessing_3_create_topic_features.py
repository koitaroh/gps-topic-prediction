import atexit
import multiprocessing as mp
from datetime import datetime, timedelta
import gensim
import geopy
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from collections import Counter
from sqlalchemy import create_engine, text


# Logging ver. 2017-10-30
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - [%(levelname)s][%(funcName)s] - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)
start_time = datetime.now()

import settings as s
import utility_database
import jpgrid
import utility_spatiotemporal_index


def create_freq_stop_list_(table_name, stoplist, conn, n=50, min_freq=1): # n was 100
    logger.info("Loading tweets to dataframe for freq stop list")
    fdist = Counter()
    sql = "Select id, words from %s where words is not NULL and x is not NULL;" % table_name
    df = pd.read_sql(text(sql), conn)
    for index, row in tqdm(df.iterrows()):
        words = row.words
        tweet_id = row.id
        words_token = gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore")
        for word in words_token:
            if word not in stoplist:
                fdist[word] += 1
    # print(f"Word frequency: {fdist}")
    common_words_reference = {word for word, freq in fdist.most_common(100)}
    logger.info(f"500 most common words: {common_words_reference}")
    common_words = {word for word, freq in fdist.most_common(n)}
    # print(f"{n} most common words: {common_words}")
    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
    stopwords = common_words.union(rare_words)
    logger.info(f'Stop words ratio to total words: {len(stopwords)}/{len(fdist)}')
    return stopwords


def create_global_topic_features(models, dictionary, tfidf, temporal_index, EXPERIMENT_PARAMETERS, topic_feature_files, table_name, stoplist, freq_stop_list, conn):
    logger.info("Loading tweets to token")
    docs_token = []
    for (i, t_index) in tqdm(enumerate(temporal_index)):  # for each t_index
        delta_time = EXPERIMENT_PARAMETERS['UNIT_TEMPORAL_TOPIC']
        query_time_start = t_index[4] - timedelta(minutes=delta_time)
        query_time_end = t_index[4] + timedelta(minutes=delta_time)
        # print(i, query_time_start, query_time_end)
        sql = "Select words from %s where (words is not NULL) and (x BETWEEN %s and %s) and (y BETWEEN %s and %s) and (tweeted_at BETWEEN '%s' and '%s');" \
              % (table_name, EXPERIMENT_PARAMETERS["AOI"][0], EXPERIMENT_PARAMETERS["AOI"][2], EXPERIMENT_PARAMETERS["AOI"][1], EXPERIMENT_PARAMETERS["AOI"][3], query_time_start, query_time_end)
        # print(sql)
        df = pd.read_sql(text(sql), conn)
        df_length = len(df.index)
        words_token_list = []
        if df_length:
            # print(df)
            for index, row in df.iterrows():  # for each tweet
                words = row.words
                words_token = gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore")
                for word in words_token:
                    if word not in stoplist and word not in freq_stop_list:
                        words_token_list.append(word)
            # print(words_token_list)
        docs_token.append(words_token_list)
    print(len(docs_token))
    # print(len(docs_doc2vec))

    logger.info("Creating corpora for LSI and LDA")
    corpus_mm = [dictionary.doc2bow(text) for text in docs_token]
    corpus_tfidf = tfidf[corpus_mm]
    # print(corpus_mm)
    # print(corpus_tfidf[0])
    # for docs in corpus_tfidf:
    #     print(docs)

    logger.info("Loading topic features.")

    logger.info("Loading LSI features")
    corpus_lsi = models["lsi"][corpus_tfidf]
    # print(corpus_lsi[0])
    # for docs in corpus_lsi:
    #     print(docs)
    topic_feature_lsi_global = gensim.matutils.corpus2dense(corpus_lsi, EXPERIMENT_PARAMETERS['num_topic_k']).T
    # print(topic_feature_lsi_global)
    # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_lsi_global.shape)
    np.save(file=topic_feature_files["lsi"], arr=topic_feature_lsi_global)

    logger.info("Loading LDA features")
    corpus_lda = models["lda"][corpus_mm]
    # print(corpus_lsi[0])
    # for docs in corpus_lsi:
    #     print(docs)
    topic_feature_lda_global = gensim.matutils.corpus2dense(corpus_lda, EXPERIMENT_PARAMETERS['num_topic_k']).T
    # print(topic_feature_lsi_global)
    # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_lda_global.shape)
    np.save(file=topic_feature_files["lda"], arr=topic_feature_lda_global)

    logger.info("Loading Doc2Vec features")
    topic_feature_doc2vec_global = np.zeros([len(temporal_index), EXPERIMENT_PARAMETERS['num_topic_k']], dtype=np.float32)
    for (i, doc) in tqdm(enumerate(docs_token)):
        # print(docs)
        docvec = models["doc2vec"].infer_vector(doc)
        # print(docvec)
        topic_feature_doc2vec_global[i, :] = docvec
    # print(topic_feature_doc2vec_global)
    # print(topic_feature_doc2vec_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_doc2vec_global.shape)
    np.save(file=topic_feature_files["doc2vec"], arr=topic_feature_doc2vec_global)

def create_small_topic_features(models, dictionary, tfidf, temporal_index, EXPERIMENT_PARAMETERS, topic_feature_files, table_name, stoplist, freq_stop_list, conn):
    logger.info("Loading tweets to token")
    docs_token = []
    for (i, t_index) in tqdm(enumerate(temporal_index)):  # for each t_index
        delta_time = EXPERIMENT_PARAMETERS['UNIT_TEMPORAL_TOPIC']
        query_time_start = t_index[4] - timedelta(minutes=delta_time)
        query_time_end = t_index[4] + timedelta(minutes=delta_time)
        # print(i, query_time_start, query_time_end)
        sql = "Select words from %s where (words is not NULL) and (x BETWEEN %s and %s) and (y BETWEEN %s and %s) and (tweeted_at BETWEEN '%s' and '%s');" \
              % (table_name, EXPERIMENT_PARAMETERS["AOI_SMALL"][0], EXPERIMENT_PARAMETERS["AOI_SMALL"][2], EXPERIMENT_PARAMETERS["AOI_SMALL"][1], EXPERIMENT_PARAMETERS["AOI_SMALL"][3], query_time_start, query_time_end)
        # print(sql)
        df = pd.read_sql(text(sql), conn)
        df_length = len(df.index)
        words_token_list = []
        if df_length:
            # print(df)
            for index, row in df.iterrows():  # for each tweet
                words = row.words
                words_token = gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore")
                for word in words_token:
                    if word not in stoplist and word not in freq_stop_list:
                        words_token_list.append(word)
            # print(words_token_list)
        docs_token.append(words_token_list)
    print(len(docs_token))
    # print(len(docs_doc2vec))

    logger.info("Creating corpora for LSI and LDA")
    corpus_mm = [dictionary.doc2bow(text) for text in docs_token]
    corpus_tfidf = tfidf[corpus_mm]
    # print(corpus_mm)
    # print(corpus_tfidf[0])
    # for docs in corpus_tfidf:
    #     print(docs)

    logger.info("Loading topic features.")

    logger.info("Loading LSI features")
    corpus_lsi = models["lsi"][corpus_tfidf]
    # print(corpus_lsi[0])
    # for docs in corpus_lsi:
    #     print(docs)
    topic_feature_lsi_global = gensim.matutils.corpus2dense(corpus_lsi, EXPERIMENT_PARAMETERS['num_topic_k']).T
    # print(topic_feature_lsi_global)
    # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_lsi_global.shape)
    np.save(file=topic_feature_files["lsi"], arr=topic_feature_lsi_global)

    logger.info("Loading LDA features")
    corpus_lda = models["lda"][corpus_mm]
    # print(corpus_lsi[0])
    # for docs in corpus_lsi:
    #     print(docs)
    topic_feature_lda_global = gensim.matutils.corpus2dense(corpus_lda, EXPERIMENT_PARAMETERS['num_topic_k']).T
    # print(topic_feature_lsi_global)
    # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_lda_global.shape)
    np.save(file=topic_feature_files["lda"], arr=topic_feature_lda_global)

    logger.info("Loading Doc2Vec features")
    topic_feature_doc2vec_global = np.zeros([len(temporal_index), EXPERIMENT_PARAMETERS['num_topic_k']], dtype=np.float32)
    for (i, doc) in tqdm(enumerate(docs_token)):
        # print(docs)
        docvec = models["doc2vec"].infer_vector(doc)
        # print(docvec)
        topic_feature_doc2vec_global[i, :] = docvec
    # print(topic_feature_doc2vec_global)
    # print(topic_feature_doc2vec_global.shape) # (num_topic:50, num_doc:4)
    logger.info("Output numpy array shape: %s", topic_feature_doc2vec_global.shape)
    np.save(file=topic_feature_files["doc2vec"], arr=topic_feature_doc2vec_global)


if __name__ == '__main__':
    # slack_client = s.sc
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    EXPERIMENT_ENVIRONMENT = s.EXPERIMENT_ENVIRONMENT
    LSI_MODEL_FILE = s.LSI_MODEL_FILE
    LDA_MODEL_FILE = s.LDA_MODEL_FILE
    DOC2VEC_MODEL_FILE = s.DOC2VEC_MODEL_FILE
    STOPLIST_FILE = s.STOPLIST_FILE
    DICT_FILE = s.DICT_FILE
    MM_CORPUS_FILE = s.MM_CORPUS_FILE
    TFIDF_FILE = s.TFIDF_FILE

    LSI_TOPIC_FILE = s.LSI_TOPIC_FILE
    LDA_TOPIC_FILE = s.LDA_TOPIC_FILE
    DOC2VEC_TOPIC_FILE = s.DOC2VEC_TOPIC_FILE

    LSI_TOPIC_SMALL_FILE = s.LSI_TOPIC_SMALL_FILE
    LDA_TOPIC_SMALL_FILE = s.LDA_TOPIC_SMALL_FILE
    DOC2VEC_TOPIC_SMALL_FILE = s.DOC2VEC_TOPIC_SMALL_FILE

    TABLE_NAME = "tweet_table_201207"

    cores = mp.cpu_count()
    logger.info("Using %s cores" % cores)
    pool = mp.Pool(cores)

    if EXPERIMENT_ENVIRONMENT == "remote":
        engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter_remote()
    elif EXPERIMENT_ENVIRONMENT == "local":
        engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter_ssh()

    temporal_index = utility_spatiotemporal_index.define_temporal_index(EXPERIMENT_PARAMETERS)
    logger.info(f"Temporal index length: {len(temporal_index)}")

    stop_list = open(STOPLIST_FILE).read().splitlines()
    logger.info(f"pre-difined stop_list: {stop_list}")
    freq_stop_list = create_freq_stop_list_(TABLE_NAME, stop_list, conn, n=10, min_freq=1)
    logger.info(f"inferred freq_stop_list: {freq_stop_list}")

    # Load models
    dictionary = gensim.corpora.Dictionary.load(DICT_FILE)
    lsi = gensim.models.LsiModel.load(LSI_MODEL_FILE)
    lda = gensim.models.LdaModel.load(LDA_MODEL_FILE)
    doc2vec = Doc2Vec.load(DOC2VEC_MODEL_FILE)
    tfidf = gensim.models.TfidfModel.load(TFIDF_FILE)


    # Create topic features
    if EXPERIMENT_ENVIRONMENT == "remote":
        engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter_remote()
    elif EXPERIMENT_ENVIRONMENT == "local":
        engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter_ssh()


    models = {"lsi": lsi, "lda": lda, "doc2vec": doc2vec}
    topic_feature_files = {"lsi": LSI_TOPIC_FILE, "lda": LDA_TOPIC_FILE, "doc2vec": DOC2VEC_TOPIC_FILE}
    create_global_topic_features(models, dictionary, tfidf, temporal_index, EXPERIMENT_PARAMETERS, topic_feature_files, TABLE_NAME, stop_list, freq_stop_list, conn)
    models = {"lsi": lsi, "lda": lda, "doc2vec": doc2vec}
    topic_feature_files = {"lsi": LSI_TOPIC_SMALL_FILE, "lda": LDA_TOPIC_SMALL_FILE, "doc2vec": DOC2VEC_TOPIC_SMALL_FILE}
    create_small_topic_features(models, dictionary, tfidf, temporal_index, EXPERIMENT_PARAMETERS, topic_feature_files, TABLE_NAME, stop_list, freq_stop_list, conn)


    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)
