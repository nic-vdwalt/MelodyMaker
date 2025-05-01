import MySQLdb

def Connect():
    db = MySQLdb.connect(host="localhost",    
                     user="root",         
                     passwd="",  
                     db="melodymakerdb")        

cursor = connect.cursor()

#user will input mood in order to search
mood=""
genre=""

sql = "SELECT * FROM tblCSV inner join mood on tblCSV.moodID=mood.moodID inner join tblgenre on tblCSV.genreID=tblgenre.genreID"

cursor.execute(sql)

results = cursor.fetchall()
for row in results:
    filePath = row[0]

cursor.close()
