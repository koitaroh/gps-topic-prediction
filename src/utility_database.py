import os
import pandas
import collections
import smart_open
import random
import multiprocessing
import sqlalchemy
import pymysql
import configparser


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


# Database configuration
conf = configparser.ConfigParser()
conf.read('config.cfg')
LOCAL_DB_POSTGRESQL = {
    "host": conf.get('local_postgresql', 'host'),
    "user": conf.get('local_postgresql', 'user'),
    "passwd": conf.get('local_postgresql', 'passwd'),
    "db_name": conf.get('local_postgresql', 'db_name'),
}

LOCAL_DB_MYSQL = {
    "host": conf.get('local_mysql', 'host'),
    "user": conf.get('local_mysql', 'user'),
    "passwd": conf.get('local_mysql', 'passwd'),
    "db_name": conf.get('local_mysql', 'db_name'),
}

ENGINE_CONF = "postgresql://" + LOCAL_DB_POSTGRESQL["user"] + ":" + LOCAL_DB_POSTGRESQL["passwd"] + "@" + LOCAL_DB_POSTGRESQL["host"] + ":5432/" + LOCAL_DB_POSTGRESQL["db_name"]
ENGINE_CONF_LOCAL_MYSQL = "mysql+pymysql://" + LOCAL_DB_MYSQL["user"] + ":" + LOCAL_DB_MYSQL["passwd"] + "@" + LOCAL_DB_MYSQL["host"] + "/" + LOCAL_DB_MYSQL["db_name"] + "?charset=utf8mb4"


def establish_db_connection_postgresql_geotweet():
    # logger.info('Establishing engine.')
    engine = sqlalchemy.create_engine(ENGINE_CONF)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata


def establish_db_connection_mysql_local():
    engine = sqlalchemy.create_engine(ENGINE_CONF_LOCAL_MYSQL)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata

#
# def establish_db_connection_postgresql_inbounds_with_ssh():
#     server = SSHTunnelForwarder(
#         ('52.69.170.146', 22), #Remote server IP and SSH port
#         ssh_username = "koitaroh",
#         ssh_pkey= "/Users/koitaroh/.ssh/miya_dev2_koitaroh.pem",
#         # ssh_password = "<password>",
#         remote_bind_address=('postgres.cxajboomlk4p.ap-northeast-1.rds.amazonaws.com', 5432))
#     #PostgreSQL server IP and sever port on remote machine
#     server.start() #start ssh sever
#     logger.info('Server connected via SSH')
#
#     #connect to PostgreSQL
#     local_port = str(server.local_bind_port)
#     engine = sqlalchemy.create_engine('postgresql://engineer:nLrBk9_5@localhost:' + local_port +'/inbounds')
#     conn = engine.connect()
#     metadata = sqlalchemy.MetaData(engine)
#
#     conn.execute("SET ROLE dev;")
#
#     return engine, conn, metadata
#
#
# def establish_db_connection_mysql_miyazawa():
#     # logger.info('Establishing engine.')
#     engine = sqlalchemy.create_engine("mysql+pymysql://" + "nl_staff" + ":" + "Rg215xst_98!" + "@" + "52.69.170.146" + "/" + "keyword_recommendation" + "?charset=utf8mb4", echo=False)
#     conn = engine.connect()
#     metadata = sqlalchemy.MetaData(engine)
#     return engine, conn, metadata
#
#
# def establish_db_connection_mysql_inbounds():
#     # logger.info('Establishing engine.')
#     engine = sqlalchemy.create_engine("mysql+pymysql://" + "inbounds_img" + ":" + "nl-ii0105" + "@" + "52.69.93.146" + "/" + "inbounds" + "?charset=utf8mb4", echo=False)
#     conn = engine.connect()
#     metadata = sqlalchemy.MetaData(engine)
#     return engine, conn, metadata
#
#
# def establish_db_connection_mysql_divided_inbounds():
#     # logger.info('Establishing engine.')
#     engine = sqlalchemy.create_engine("mysql+pymysql://" + "inbounds_img" + ":" + "nl-ii0105" + "@" + "52.69.93.146" + "/" + "divided_inbounds" + "?charset=utf8mb4", echo=False)
#     conn = engine.connect()
#     metadata = sqlalchemy.MetaData(engine)
#     return engine, conn, metadata


if __name__ == '__main__':
    engine, conn, metadata = establish_db_connection_postgresql_geotweet(ENGINE_CONF)