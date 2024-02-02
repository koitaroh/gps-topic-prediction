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
    sql = f"SELECT id, words FROM {table_name} WHERE words is not NULL and x is not NULL;"
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


def load_tweets_to_dataframe_from_database(table_name, stoplist, freq_stop_list, conn):
    logger.info("Loading tweets to dataframe")
    docs_token = []
    docs_doc2vec = []
    sql = f"Select id, words from {table_name} where words is not NULL and x is not NULL;"
    df = pd.read_sql(text(sql), conn)
    df_length = len(df)
    for index, row in tqdm(df.iterrows()):
        words = row.words
        tweet_id = row.id
        words_token = gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore")
        words_token_list = []
        for word in words_token:
            if word not in stoplist and word not in freq_stop_list:
                words_token_list.append(word)
        docs_doc2vec.append(TaggedDocument(words=words_token_list, tags=[tweet_id]))
        # docs.append(x for x in gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)
        # yield (x for x in gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore") if x not in stoplist)
        docs_token.append(words_token_list)
    return docs_token, docs_doc2vec


def create_corpora(token, dict_file_path, mm_corpora_file, tfidf_file):
    logger.info("Creating corpora")

    with open(dict_file_path, 'wb') as dict_file:
        dictionary = gensim.corpora.Dictionary(token)
        dictionary.save(dict_file)
        print(dictionary)
        # print(dictionary.token2id)
        # print(token)

        corpus_mm = [dictionary.doc2bow(text) for text in token]
        gensim.corpora.MmCorpus.serialize(mm_corpora_file, corpus_mm)
        tfidf = gensim.models.TfidfModel(corpus_mm)
        tfidf.save(tfidf_file)
        corpus_tfidf = tfidf[corpus_mm]

        # print(corpus_mm)
        # print(corpus_tfidf)

    return dictionary, corpus_mm, corpus_tfidf, tfidf


def train_lsi_model(dictionary, corpus_tfidf, lsi_model_file, EXPERIMENT_PARAMETERS):
    logger.info('Training LSI')
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=EXPERIMENT_PARAMETERS['num_topic_k'])  # initialize an LSI transformation
    lsi.save(lsi_model_file)  # same for tfidf, lda, ...
    # print(lsi[corpus_tfidf])
    # corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    print(lsi.show_topics(50))
    return lsi


def train_lda_model(dictionary, corpus_mm, lda_model_file, EXPERIMENT_PARAMETERS):
    logger.info('Training LDA')
    lda = gensim.models.LdaModel(corpus_mm, id2word=dictionary, num_topics=EXPERIMENT_PARAMETERS['num_topic_k'], alpha='auto')
    lda.save(lda_model_file)  # same for tfidf, lda, ...
    # print(lda[corpus_mm])
    # corpus_lda = lda[corpus_mm]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    print(lda.show_topics(50, formatted=True))
    return lda


def train_doc2vec_model(docs_doc2vec, doc2vec_model_file, cores, EXPERIMENT_PARAMETERS):
    logger.info("Training Doc2Vec")
    # PV-DBOW
    # model_dbow = Doc2Vec(docs_doc2vec, dm=0, dbow_words=1, size=50, window=15, min_count=10, iter=10, workers=cores)
    # model_dbow.save(doc2vec_model_file)
    # PV-DM w/average
    model_dm = Doc2Vec(docs_doc2vec, dm=1, dm_mean=1, vector_size=EXPERIMENT_PARAMETERS['num_topic_k'], window=8, min_count=10, epochs=10, workers=cores)
    model_dm.save(doc2vec_model_file)
    return model_dm


# def create_global_topic_features(models, dictionary, tfidf, temporal_index, EXPERIMENT_PARAMETERS, topic_feature_files, table_name, stoplist, freq_stop_list, conn):
#     logger.info("Loading tweets to token")
#     docs_token = []
#     for (i, t_index) in tqdm(enumerate(temporal_index)):  # for each t_index
#         delta_time = EXPERIMENT_PARAMETERS['UNIT_TEMPORAL_TOPIC']
#         query_time_start = t_index[4] - timedelta(minutes=delta_time)
#         query_time_end = t_index[4] + timedelta(minutes=delta_time)
#         # print(i, query_time_start, query_time_end)
#         sql = "Select words from %s where (words is not NULL) and (x BETWEEN %s and %s) and (y BETWEEN %s and %s) and (tweeted_at BETWEEN '%s' and '%s');" \
#               % (table_name, EXPERIMENT_PARAMETERS["AOI"][0], EXPERIMENT_PARAMETERS["AOI"][2], EXPERIMENT_PARAMETERS["AOI"][1], EXPERIMENT_PARAMETERS["AOI"][3], query_time_start, query_time_end)
#         # print(sql)
#         df = pd.read_sql(sql, conn)
#         df_length = len(df.index)
#         words_token_list = []
#         if df_length:
#             # print(df)
#             for index, row in df.iterrows():  # for each tweet
#                 words = row.words
#                 words_token = gensim.utils.tokenize(words, lowercase=True, deacc=True, errors="ignore")
#                 for word in words_token:
#                     if word not in stoplist and word not in freq_stop_list:
#                         words_token_list.append(word)
#             # print(words_token_list)
#         docs_token.append(words_token_list)
#     print(len(docs_token))
#     # print(len(docs_doc2vec))

#     logger.info("Creating corpora for LSI and LDA")
#     corpus_mm = [dictionary.doc2bow(text) for text in docs_token]
#     corpus_tfidf = tfidf[corpus_mm]
#     # print(corpus_mm)
#     # print(corpus_tfidf[0])
#     # for docs in corpus_tfidf:
#     #     print(docs)

#     logger.info("Loading topic features.")

#     logger.info("Loading LSI features")
#     corpus_lsi = models["lsi"][corpus_tfidf]
#     # print(corpus_lsi[0])
#     # for docs in corpus_lsi:
#     #     print(docs)
#     topic_feature_lsi_global = gensim.matutils.corpus2dense(corpus_lsi, EXPERIMENT_PARAMETERS['num_topic_k']).T
#     # print(topic_feature_lsi_global)
#     # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
#     logger.info("Output numpy array shape: %s", topic_feature_lsi_global.shape)
#     np.save(file=topic_feature_files["lsi"], arr=topic_feature_lsi_global)

#     logger.info("Loading LDA features")
#     corpus_lda = models["lda"][corpus_mm]
#     # print(corpus_lsi[0])
#     # for docs in corpus_lsi:
#     #     print(docs)
#     topic_feature_lda_global = gensim.matutils.corpus2dense(corpus_lda, EXPERIMENT_PARAMETERS['num_topic_k']).T
#     # print(topic_feature_lsi_global)
#     # print(topic_feature_lsi_global.shape) # (num_topic:50, num_doc:4)
#     logger.info("Output numpy array shape: %s", topic_feature_lda_global.shape)
#     np.save(file=topic_feature_files["lda"], arr=topic_feature_lda_global)

#     logger.info("Loading Doc2Vec features")
#     topic_feature_doc2vec_global = np.zeros([len(temporal_index), EXPERIMENT_PARAMETERS['num_topic_k']], dtype=np.float32)
#     for (i, doc) in tqdm(enumerate(docs_token)):
#         # print(docs)
#         docvec = models["doc2vec"].infer_vector(doc)
#         # print(docvec)
#         topic_feature_doc2vec_global[i, :] = docvec
#     # print(topic_feature_doc2vec_global)
#     # print(topic_feature_doc2vec_global.shape) # (num_topic:50, num_doc:4)
#     logger.info("Output numpy array shape: %s", topic_feature_doc2vec_global.shape)
#     np.save(file=topic_feature_files["doc2vec"], arr=topic_feature_doc2vec_global)




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
    TABLE_NAME = EXPERIMENT_PARAMETERS['TWEET_TABLE_NAME']

    cores = mp.cpu_count()
    logger.info("Using %s cores" % cores)
    pool = mp.Pool(cores)

    engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter(EXPERIMENT_ENVIRONMENT)

    temporal_index = utility_spatiotemporal_index.define_temporal_index(EXPERIMENT_PARAMETERS)
    logger.info(f"Temporal index length: {len(temporal_index)}")

    stop_list = open(STOPLIST_FILE).read().splitlines()
    logger.info(f"pre-difined stop_list: {stop_list}")
    freq_stop_list = create_freq_stop_list_(TABLE_NAME, stop_list, conn, n=10, min_freq=1)
    logger.info(f"inferred freq_stop_list: {freq_stop_list}")

    # Load tweets for LSI, LDA
    engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter(EXPERIMENT_ENVIRONMENT)
    docs_token, docs_doc2vec = load_tweets_to_dataframe_from_database(TABLE_NAME, stop_list, freq_stop_list, conn)
    dictionary, corpus_mm, corpus_tfidf, tfidf = create_corpora(docs_token, DICT_FILE, MM_CORPUS_FILE, TFIDF_FILE)

    # Train models
    lsi = train_lsi_model(dictionary, corpus_tfidf, LSI_MODEL_FILE, EXPERIMENT_PARAMETERS)
    lda = train_lda_model(dictionary, corpus_mm, LDA_MODEL_FILE, EXPERIMENT_PARAMETERS)
    doc2vec = train_doc2vec_model(docs_doc2vec, DOC2VEC_MODEL_FILE, cores, EXPERIMENT_PARAMETERS)

    elapsed = str((datetime.now() - start_time))
    logger.info(f"Finished in: {elapsed}")
    atexit.register(s.exit_handler, __file__, elapsed)
