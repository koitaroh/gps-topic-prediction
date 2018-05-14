CREATE TABLE purpose_3 as
SELECT t1.uri, t1.userid, t1.time_start, t1.time_end, t2.id as purpose_id, t1.purpose as purpose_ja, t2.purpose_en
FROM purpose_2 as t1
INNER JOIN purpose_master as t2 ON t1.purpose = t2.purpose