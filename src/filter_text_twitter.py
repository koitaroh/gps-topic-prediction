# Last Update: 2017-08-04
# @author: Satoshi Miyazawa
# koitaroh@gmail.com
# Specify a table on a database. Apply filter_japanese_text.py on corresponding column.

import pymysql
import sqlalchemy

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

from filter_japanese_text import filter_japanese_text_mecab
# import settings as s
import utility_database


def prepare_tweet_db(table_name, engine, conn, metadata):
    table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=engine)
    s = sqlalchemy.select([table.c.id, table.c.text]).where(table.c.x != None).where(table.c.text != None).where(table.c.lang == "en")
    result = conn.execute(s)
    for row in result.fetchall():
        id = row['id']
        text = row['text']
        try:
            words = filter_japanese_text_mecab(text)
        except Exception as err:
            logger.error(err)
            words = ""
            pass
        logger.debug(words)
        stmt = table.update().where(table.c.id == id).values(words=words)
        conn.execute(stmt)
    result.close()
    return None


if __name__ == '__main__':
    # TABLE_NAME = "tweet_table_201207"
    TABLE_NAME = "tweet_table_201207"
    engine, conn, metadata = utility_database.establish_db_connection_mysql_local()

    # test1 = "帰るよー (@ 渋谷駅 (Shibuya Sta.) in 渋谷区, 東京都) https://t.co/UwXP9Gr0wJ check this out (http://t.co/nYHbleBtBT)"
    # test2 = "終電変わらず(｀・ω・´)ゞ @ 川崎駅にタッチ！ http://t.co/DJFKEEUW3n"
    # test3 = "月ザンギョ100とか200とかそら死ぬわな、と実感しつつある。 (@ ノロワレハウス II in 杉並区, 東京都) https://t.co/BkcI7uNigi"

    prepare_tweet_db(TABLE_NAME, engine, conn, metadata)


