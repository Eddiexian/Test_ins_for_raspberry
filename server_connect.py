################################################
# 與server連接的ip, id, password在config.py更改 #
################################################
import os
import ftplib
import pymysql
import pyodbc
import config

current_path = os.getcwd()

def deletefile(file_name):
    # 刪除local端影片檔
    try:
        os.remove(os.path.join(current_path, file_name))
    except OSError as e:
        print(e)
    else:
        print('file is deleted successfully')

def insertDB(sqlstring):
    # 結果回傳DB

    conn = pyodbc.connect(driver='{FreeTDS}',
    server= config.DB_ip + '\\SQLEXPRESS',
    uid=config.DB_id, pwd=config.DB_password)
    
    cursor = conn.cursor()
    cursor.execute("select * from [TEST_Ins_record_db].[dbo].[detect_info]")
    print('Write data to database successfully!')
    cursor.execute(sqlstring)
    conn.commit()
    
    # MySQL測試用
#   conn = pymysql.connect(host=DB_ip, user=DB_id, passwd=DB_password, db='PI' )
#   cursor = conn.cursor()
#   cursor.execute(sqlstring)
#   conn.commit()

def upload2ftp(file_name):

    f = ftplib.FTP(config.FTP_ip,timeout=20000)
    f.login(config.FTP_id, config.FTP_password)
    file_remote = config.FTP_file_path + file_name
    file_local = os.path.join(current_path, file_name)
    bufsize = 1024
    fp = open(file_local,'rb')
    f.storbinary('STOR ' + file_remote, fp, bufsize)
    fp.close()
    deletefile(file_name)

def insertSQL(file_name, status,machine_id,t_left_times,t_right_times,durations,reason):
    sql = '''
    insert into [TEST_Ins_record_db].[dbo].[detect_info]
    (record_time, mfg_d, mfg_h, status, file_path, machine_loc,t_left_times,t_right_times,durations,reason)
    values(getdate(), CONVERT(VARCHAR, dateadd(HOUR, -7, getdate()),23),datepart(hh,getdate()),'{}','{}','{}',{},{},{},{})
    '''.format(status, file_name,machine_id,t_left_times,t_right_times,durations,reason)
    print(sql)
    insertDB(sql)
