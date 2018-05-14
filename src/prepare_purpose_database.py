import requests
import json
import csv
import os
import time
import sqlalchemy
from sqlalchemy import between, func
from sshtunnel import SSHTunnelForwarder #Run pip install sshtunnel
from sqlalchemy.orm import sessionmaker #Run pip install sqlalchemy
import utility_database


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


def prepare_purpose_database(purpose_csv_file, table_name_purpose):
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
        table_purpose = sqlalchemy.Table(table_name_purpose, metadata, autoload=True, autoload_with=engine)

        with open(purpose_csv_file) as purpose_csv:
            reader = csv.reader(purpose_csv)
            next(reader, None)
            for row in reader:
                username_text = row[1]
                logger.info("Username: %s" % username_text)

                days = ["07-24", "07-25", "07-26", "07-27", "07-28", "07-29", "07-30", "07-31"]
                hours = ["00", "03", "06", "09", "12", "15", "18", "21"]
                column_num = 9
                for day in days:
                    for hour in hours:
                        time_start = "2016-" + day + " " + hour + ":00:00"
                        time_end = "2016-" + day + " " + str(int(hour) + 2) + ":59:59"
                        stmt = table_purpose.insert().values(userid=username_text, time_start=time_start, time_end=time_end, purpose=row[column_num])
                        conn.execute(stmt)
                        column_num += 1
    conn.close()
    return None


def iterate_purpose_table(engine, conn, metadata, table_name_purpose, table_name_trajectory):

    table_purpose = sqlalchemy.Table(table_name_purpose, metadata, autoload=True, autoload_with=engine)
    table_trajectory = sqlalchemy.Table(table_name_trajectory, metadata, autoload=True, autoload_with=engine)

    select_purpose = sqlalchemy.select([table_purpose])
    result_purpose = conn.execute(select_purpose)
    for row in result_purpose.fetchall():
        uri = row[0]
        username = row[1]
        time_start = row[2]
        time_end = row[3]
        purpose = row[4]
        # logger.info("Applying purpose for user: %s between %s and %s as purpose %s" % (username, time_start, time_end, purpose))
        apply_purpose_to_trajectory(conn, table_trajectory, uri, time_start, time_end, purpose)
    result_purpose.close()
    conn.close()

    return None


def apply_purpose_to_trajectory(conn, table_trajectory, uri, time_start, time_end, purpose):
    stmt_update_trajectory = table_trajectory.update().where(table_trajectory.c.user_uri == uri).where(between(table_trajectory.c.time, time_start, time_end)).where(table_trajectory.c.purpose == None).values(purpose=purpose)
    conn.execute(stmt_update_trajectory)
    return None


if __name__ == '__main__':
    TABLE_NAME_PURPOSE = "purpose_3"
    TABLE_NAME_TRAJECTORY = "gps_log_2"

    TIMESTART = "2016–05–03"
    TIMEEND = "2016–07–31"

    engine, conn, metadata = utility_database.establish_db_connection_postgresql_geotweet()

    PURPOSE_CSV_FILE = "/Users/koitaroh/Documents/GitHub/Workspace/src/data/情報銀行プロジェクト実験(2016-07 第4週) 補足情報入力フォーム (Responses) - Form Responses 1.csv"
    # prepare_purpose_database(PURPOSE_CSV_FILE, TABLE_NAME_PURPOSE)
    iterate_purpose_table(engine, conn, metadata, TABLE_NAME_PURPOSE, TABLE_NAME_TRAJECTORY)

