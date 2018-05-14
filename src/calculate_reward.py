import requests
import json
import csv
import os
import time
import sqlalchemy
from sqlalchemy import between, func
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

# iterate user
# iterate date
# calculate reward
# export csv

def iterate_user(reward_csv_dir, users_csv_file, date_csv_file, table_name_tweet, table_name_gps):


    with SSHTunnelForwarder(
        ('ibank.iis.u-tokyo.ac.jp', 22), #Remote server IP and SSH port
        ssh_username = "geotweet",
        ssh_pkey= "/Users/koitaroh/.ssh/id_rsa",
        # ssh_password = "<password>",
        remote_bind_address=('localhost', 5432)) as server: #PostgreSQL server IP and sever port on remote machine
        server.start() #start ssh sever
        # logger.info('Server connected via SSH')

        #connect to PostgreSQL
        local_port = str(server.local_bind_port)
        engine = sqlalchemy.create_engine('postgresql://geotweet:D3vsZuAi@localhost:' + local_port +'/geotweet')
        conn = engine.connect()
        metadata = sqlalchemy.MetaData(engine)
        table_tweet = sqlalchemy.Table(table_name_tweet, metadata, autoload=True, autoload_with=engine)
        table_gps = sqlalchemy.Table(table_name_gps, metadata, autoload=True, autoload_with=engine)

        username_text = ''
        username_uri = ''
        with open(users_csv_file) as users_reward_csv:
            reader = csv.reader(users_reward_csv)
            for row in reader:
                username_text = row[0]
                logger.info("Username: %s" % username_text)
                username_uri = row[1]

        # username_text_test = "ae101silk11"
        # username_uri_test = "geotweet:user:0e750dfc-d7d4-4468-a8d6-ae0d4bd21ac7"
        # username_uri = username_uri_test
        # username_text = username_text_test

                with open(os.path.join(reward_csv_dir, "rewards_" + username_text+".csv"), 'w', newline='') as reward_csv:
                    writer = csv.writer(reward_csv)
                    writer.writerow(["date", "num_tweet", "reward_tweet", "num_gps", "reward_gps", "reward_total"])
                    reward_list = iterate_date(date_csv_file, username_uri, table_tweet, table_gps, conn)
                    writer.writerows(reward_list)
    conn.close()

    return None



def iterate_date(date_csv_file, username_uri, table_tweet, table_gps, conn):
    reward_list = []
    with open(date_csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            date = row[0]
            num_tweet, reward_tweet, num_gps, reward_gps, reward_total = calculate_reward(username_uri, date, table_tweet, table_gps, conn)
            reward_list.append([date, num_tweet, reward_tweet, num_gps, reward_gps, reward_total])
    return reward_list


def calculate_reward(username_uri, date, table_tweet, table_gps, conn):

    date_start = date + " 00:00:00"
    date_end = date + " 23:59:59"

    reward_tweet = 0
    reward_gps = 0
    rewward_total = 0
    num_tweet = 0
    num_gps = 0

    # with SSHTunnelForwarder(
    #     ('ibank.iis.u-tokyo.ac.jp', 22), #Remote server IP and SSH port
    #     ssh_username = "geotweet",
    #     ssh_pkey= "/Users/koitaroh/.ssh/id_rsa",
    #     # ssh_password = "<password>",
    #     remote_bind_address=('localhost', 5432)) as server: #PostgreSQL server IP and sever port on remote machine
    #     server.start() #start ssh sever
    #     # logger.info('Server connected via SSH')
    #
    #     #connect to PostgreSQL
    #     local_port = str(server.local_bind_port)
    #     engine = sqlalchemy.create_engine('postgresql://geotweet:D3vsZuAi@localhost:' + local_port +'/geotweet')
    #     conn = engine.connect()
    #     metadata = sqlalchemy.MetaData(engine)
    #     table_tweet = sqlalchemy.Table(table_name_tweet, metadata, autoload=True, autoload_with=engine)
    #     table_gps = sqlalchemy.Table(table_name_gps, metadata, autoload=True, autoload_with=engine)

    # logger.info('Selecting tweets')
    select_tweet = sqlalchemy.select([func.count(table_tweet.c.tweet_id)]).where(
        table_tweet.c.user_uri == username_uri).where(between(table_tweet.c.time, date_start, date_end))
    result_tweet = conn.execute(select_tweet)
    for row in result_tweet.fetchall():
        num_tweet = row[0]
        # logger.info("Number of tweets: %s" % row)
        if row[0] > 0:
            reward_tweet = 1
    result_tweet.close()

    # logger.info('Selecting GPS')
    select_gps = sqlalchemy.select([func.count(table_gps.c.time)]).where(
        table_gps.c.user_uri == username_uri).where(between(table_gps.c.time, date_start, date_end))
    result_gps = conn.execute(select_gps)
    for row in result_gps.fetchall():
        num_gps = row[0]
        # logger.info("Number of gps: %s" % row)
        if row[0] > 36:
            reward_gps = 1
    result_gps.close()

    reward_total = reward_tweet and reward_gps

    logger.info("User %s date: %s tweet: %s -> %s gps: %s -> %s total: %s" % (username_uri, date, num_tweet, reward_tweet, num_gps, reward_gps, reward_total))

        #
        # for root, dirs, files in os.walk(csv_dir):
        #     for fn in files:
        #         logger.info(fn)
        #         csv_file = root + os.sep + fn
        #         # logger.info(csv_file)
        #         insert_list_dict = csv_to_list(csv_file)
        #         ins = table.insert()
        #         conn.execute(ins, insert_list_dict)

        # for row in result_tweet.fetchall():
        #     logger.info(row)
        #     # logger.info("# of tweets: %s" % len(row))
        #     # id_str = row['id_str']
        #     # text = row['text']
        #     # words = filter_japanese_text(text)
        #     # logger.debug(words)
        #     # stmt = table.update().where(table.c.id_str == id_str).values(words=words)
        #     # conn.execute(stmt)
        # result_tweet.close()


    return num_tweet, reward_tweet, num_gps, reward_gps, reward_total


if __name__ == '__main__':
    TEST_USER_TEXT = "toshimabou"
    # TEST_USER = "geotweet:user:ae0db06d-12ec-4ccf-aee6-959000739ad0"
    TEST_USER_URI = "geotweet:user:c86f91d0-64bc-4972-9d18-aecf2a39ba30"

    TABLE_NAME_TWEET = "tweet_log_2"
    TABLE_NAME_GPS = "gps_log"

    DATE_CSV_FILE = "/Users/koitaroh/Documents/GitHub/Workspace/src/date.csv"
    USERS_CSV_FILE = "/Users/koitaroh/Documents/Data/iBank_Experiments/username_20161028.csv"

    # USERS_REWARD_CSV_FILE = "/Users/koitaroh/Documents/Data/iBank_Experiments/reward.csv"
    REWARD_CSV_DIR = "/Users/koitaroh/Documents/Data/iBank_Experiments/rewards/"


    TIMESTART = "2016–05–03"
    TIMEEND = "2016–07–31"

    # calculate_reward(TEST_USER, "2016-07-01", TABLE_NAME_TWEET, TABLE_NAME_GPS)
    # iterate_date(DATE_CSV_FILE, TEST_USER_URI, TABLE_NAME_TWEET, TABLE_NAME_GPS)
    iterate_user(REWARD_CSV_DIR, USERS_CSV_FILE, DATE_CSV_FILE, TABLE_NAME_TWEET, TABLE_NAME_GPS)
    # csv_to_list(TEST_CSV)