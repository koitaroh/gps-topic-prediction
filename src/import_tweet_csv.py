import requests
import json
import csv
import os
import time
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


def import_csv(in_table, csv_dir):
    with SSHTunnelForwarder(
        ('ibank.iis.u-tokyo.ac.jp', 22), #Remote server IP and SSH port
        ssh_username = "geotweet",
        ssh_pkey= "/Users/koitaroh/.ssh/id_rsa",
        # ssh_password = "<password>",
        remote_bind_address=('localhost', 5432)) as server: #PostgreSQL server IP and sever port on remote machine
        server.start() #start ssh sever
        logger.info('Server connected via SSH')

        #connect to PostgreSQL
        local_port = str(server.local_bind_port)
        engine = sqlalchemy.create_engine('postgresql://geotweet:D3vsZuAi@localhost:' + local_port +'/geotweet')
        conn = engine.connect()
        metadata = sqlalchemy.MetaData(engine)
        table = sqlalchemy.Table(in_table, metadata, autoload=True, autoload_with=engine)

        for root, dirs, files in os.walk(csv_dir):
            for fn in files:
                logger.info(fn)
                csv_file = root + os.sep + fn
                # logger.info(csv_file)
                insert_list_dict = csv_to_list(csv_file)
                ins = table.insert()
                conn.execute(ins, insert_list_dict)

        conn.close()

        return None

def csv_to_list(CSV_FILE):
    out_list = []
    # logger.info(CSV_FILE)
    username = CSV_FILE.rsplit("/", maxsplit=1)[1][7:].split(".")[0]  # extract username
    # logger.info(username)
    if CSV_FILE[0] != '.':
        with open(CSV_FILE) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                tweet_id = row[0]
                timestamp = row[1]
                text = row[2]
                # logger.info(row)
                out_list.append({"name": username, "time": timestamp, "text": text, "tweet_id": tweet_id})
    # logger.info(out_list)
    return out_list


if __name__ == '__main__':
    CSV_DIR = "/Users/koitaroh/Documents/Data/iBank_Experiments/tweets/"
    TEST_CSV = "/Users/koitaroh/Documents/Data/iBank_Experiments/tweets_test/tweets_3_fooly.csv"
    TABLE_NAME = "tweet_log"
    import_csv(TABLE_NAME, CSV_DIR)
    # csv_to_list(TEST_CSV)