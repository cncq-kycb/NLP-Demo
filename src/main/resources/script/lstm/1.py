import MySQLdb
import sys

record_id=  sys.argv[1]

HOST = 'cdb-mzvws756.cd.tencentcdb.com'
USER = 'nlp'
PASSWORD = 'iop890*()'
DATABASE = 'nlp'
PORT = 10143

db = MySQLdb.connect(host=HOST, user=USER,password=PASSWORD,db=DATABASE,port=PORT, charset='utf8')
cursor = db.cursor()
sql = 'UPDATE record SET result = "1" WHERE record_id = ' + record_id
cursor.execute(sql)
db.commit()
db.close()
print('成功')