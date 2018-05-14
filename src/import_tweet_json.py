import requests
import json
import io
import csv
import json
import os
import time
import calendar
import pandas as pd
import sqlalchemy
from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy

# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)

import utility_database


def load_json_files_to_database(DATA_DIR, table_name, engine, conn, metadata):
    table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=engine)

    for root, dirs, files in os.walk(DATA_DIR):
        files.sort()
        for fn in files:
            if fn[0] != '.':
                json_file = root + fn
                load_json_file_to_database(json_file, table, engine, conn, metadata)
    return None


def load_json_file_to_database(JSON_FILE, table, engine, conn, metadata):
    logger.info("Loading JSON %s to dataframe" % JSON_FILE)
    with open(JSON_FILE, 'r', encoding='utf-8-sig') as json_file:
        tweets_dict = {}
        i = 0
        for line in json_file:
            # print(i)
            try:
                tweets_dict[i] = json.loads(line)
            except Exception as err:
                # logger.error(err)
                pass
            i += 1
        # print(i)
        # print(json.dumps(tweets_dict[i], sort_keys=True, indent=4))
        logger.info("importing %s tweets" % i)
        for key, tweet in tweets_dict.items():
            tweet_id = None
            tweeted_at = None
            user_name = ""
            user_id = None
            x = None
            y = None
            text = ""
            lang = None

            tweet_id = tweet["id"]
            tweeted_at = YmdHMS(tweet['created_at'])
            user_name = tweet["user"]["screen_name"]
            user_id = tweet['user']['id_str']
            if tweet['geo'] is not None:
                x = tweet['geo']['coordinates'][1]
                y = tweet['geo']['coordinates'][0]
            text = filter(tweet['text'])
            lang = tweet["user"]["lang"] #tweet["lang"]
            insert_list = [{"tweet_id": tweet_id, "tweeted_at": tweeted_at, "user_name": user_name, "user_id": user_id, "x": x, "y": y, "text": text, "lang": lang}]
            insert_to_table(table, engine, conn, metadata, insert_list)
    return None

# Function to convert "created at" in GMT to JST
def YmdHMS(created_at):
    time_utc = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
    unix_time = calendar.timegm(time_utc)
    time_local = time.localtime(unix_time)
    return str(time.strftime("%Y-%m-%d %H:%M:%S", time_local))

def filter(text):
    try:
        if "RT " in text:
            text = text.split(":", 1)[1]
        if text[0] == "@":
            text = text.split(" ", text.count("@"))[-1]
        if "#" in text:
            text = text.split("#", 1)[0]
        if "http" in text:
            text = text.split("http", 1)[0]
        text = text.replace('\n','')
        text = text.replace('\r','')
        text = text.replace("\'",' ')
        text = text.replace("\"",' ')
        text = text.replace("\\",' ')
        text = text.rstrip()
    except Exception as err:
        pass
        # logger.error(err)  # __str__ allows args to be printed directly,
    return text


def insert_to_table(table, engine, conn, metadata, insert_list):
    ins = table.insert()
    conn.execute(ins, insert_list)
    return None


if __name__ == '__main__':
    # JSON_DIR = "/Users/koitaroh/Documents/Data/TwitterKirimura/Tweets2012Jul_sample/"
    JSON_DIR = "/Users/koitaroh/Documents/Data/TwitterKirimura/Tweets2012Jul/"
    TABLE_NAME = "tweet_table_201207"

    engine, conn, metadata = utility_database.establish_db_connection_mysql_local()
    load_json_files_to_database(JSON_DIR, TABLE_NAME, engine, conn, metadata)