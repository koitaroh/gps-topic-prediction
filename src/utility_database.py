import sqlalchemy
import configparser
import sshtunnel

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


# Database configuration
conf = configparser.ConfigParser()
conf.read('config.cfg')


SSH_456 = {
    "host": conf.get('ssh_456', 'host'),
    "user": conf.get('ssh_456', 'user'),
    "key_path": conf.get('ssh_456', 'key_path'),
}

DB_456_GEOTWEET = {
    "host": conf.get('456_geotweet', 'host'),
    "user": conf.get('456_geotweet', 'user'),
    "passwd": conf.get('456_geotweet', 'passwd'),
    "db_name": conf.get('456_geotweet', 'db_name'),
}

DB_456_TWITTER = {
    "host": conf.get('456_mysql', 'host'),
    "user": conf.get('456_mysql', 'user'),
    "passwd": conf.get('456_mysql', 'passwd'),
    "db_name": conf.get('456_mysql', 'db_name'),
}


def establish_db_connection_postgresql_geotweet_ssh():
    server = sshtunnel.SSHTunnelForwarder(
        (SSH_456['host'], 22),
        ssh_username = SSH_456['user'],
        ssh_pkey= SSH_456['key_path'],
        remote_bind_address=('localhost', 5432))
    server.start() #start ssh sever
    logger.info('Server connected via SSH')
    local_port = str(server.local_bind_port)
    # print(local_port)
    ENGINE_CONF = "postgresql://" + DB_456_GEOTWEET["user"] + ":" + DB_456_GEOTWEET["passwd"] + "@" + DB_456_GEOTWEET["host"] + ":" + local_port + "/" + DB_456_GEOTWEET["db_name"]
    engine = sqlalchemy.create_engine(ENGINE_CONF)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata


def establish_db_connection_postgresql_geotweet_remote():
    ENGINE_CONF = "postgresql://" + DB_456_GEOTWEET["user"] + ":" + DB_456_GEOTWEET["passwd"] + "@host.docker.internal:5432/" + DB_456_GEOTWEET["db_name"]
    engine = sqlalchemy.create_engine(ENGINE_CONF)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata


def establish_db_connection_mysql_twitter_ssh():
    server = sshtunnel.SSHTunnelForwarder(
        (SSH_456['host'], 22),
        ssh_username = SSH_456['user'],
        ssh_pkey= SSH_456['key_path'],
        remote_bind_address=('localhost', 3306))
    server.start() #start ssh sever
    logger.info('Server connected via SSH')
    local_port = str(server.local_bind_port)
    # print(local_port)
    ENGINE_CONF = "mysql+pymysql://" + DB_456_TWITTER["user"] + ":" + DB_456_TWITTER["passwd"] + "@" + DB_456_TWITTER["host"] + ":" + local_port +"/" + DB_456_TWITTER["db_name"] + "?charset=utf8mb4"
    engine = sqlalchemy.create_engine(ENGINE_CONF, echo=False)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata


def establish_db_connection_mysql_twitter_remote():
    ENGINE_CONF = "mysql+pymysql://" + DB_456_TWITTER["user"] + ":" + DB_456_TWITTER["passwd"] + "@host.docker.internal:3306/" + DB_456_TWITTER["db_name"] + "?charset=utf8mb4"
    engine = sqlalchemy.create_engine(ENGINE_CONF, echo=False)
    conn = engine.connect()
    metadata = sqlalchemy.MetaData(engine)
    return engine, conn, metadata


if __name__ == '__main__':
    engine, conn, metadata = establish_db_connection_mysql_twitter_ssh()