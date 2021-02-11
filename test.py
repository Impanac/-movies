from flask import Flask, render_template,request,make_response
import plotly
import plotly.graph_objs as go
import mysql.connector
from mysql.connector import Error
import sys

import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import matplotlib.pyplot as plt
from mf import MF
import random
import math
import time
from mf import MF
from acc import Processor
        
import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

pdata=[]
connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%TOY%')order by UserID, dat ASC"
cursor = connection.cursor()
cursor.execute(sql_select_Query)
data = cursor.fetchall()
print(data)
pdata.clear()

for i in range(len(data)):
    searchuserlist=[]
    searchuserlist.append(data[i][0])
    searchuserlist.append(data[i][1])
    searchuserlist.append(data[i][2])
    '''searchuserlist.append(data[i][3])
    searchuserlist.append(data[i][4])
    searchuserlist.append(data[i][7])'''
    #temppdata.append(data[i][6])
    pdata.append(searchuserlist)

print(pdata)    
#erlyrvw = len(pdata)
#erlyrvw = erlyrvw /10
#erlyrvw=int(erlyrvw)

prvsid=''
rvw = len(pdata)
for i in range(rvw):
    curntid=str(pdata[i][0])
    print(curntid)
    print(prvsid)
    if(prvsid!=curntid):
        query ="update tags set Rtype='Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
        print(query)
        cursor.execute(query)
        connection.commit()
        prvsid=curntid
    else:
        query ="update tags set Rtype='Non-Early Reviewer',Spam='Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
        print(query)

        cursor.execute(query)
        connection.commit()
     
'''
    for i in range(erlyrvw):
        query ="update tags set Rtype='Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
        #print(query)
    
        cursor.execute(query)
        connection.commit()
    nerlyrvw=erlyrvw  
    for i in range(nerlyrvw,len(pdata)):
        query ="update tags set Rtype='Non-Early Reviewer',Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"'"
        #print(query)
    
        cursor.execute(query)
        connection.commit()



    sqlquery="UPDATE tags T1 JOIN (SELECT UserID FROM tags GROUP BY UserID HAVING count(UserID) > 1) dup ON T1.UserID = dup.UserID SET T1.Spam = 'Spam'"
    cursor.execute(sqlquery)
    connection.commit()
    
    sql_select_Query1="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE Spam='Spam' ORDER by dat asc"
    #print("hello")
    #print(sql_select_Query1)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query1)
    
    data = cursor.fetchall()
    #print(data)
    #temppdata[]
    pdata.clear()

    for i in range(len(data)):
        searchuserlist1=[]
        searchuserlist1.append(data[i][0])
        searchuserlist1.append(data[i][1])
        searchuserlist1.append(data[i][2])
        #temppdata.append(data[i][6])
        pdata.append(searchuserlist1)

    #print(pdata)    
    spamdet = len(pdata)
    spamdet = spamdet /2
    spamdet=int(spamdet)
    #print(spamdet)

    for i in range(spamdet):
        query ="update tags set Spam='Non-Spam' where MovieID='"+str(pdata[i][1])+"' and UserID='"+str(pdata[i][0])+"' and Tag='"+str(pdata[i][2])+"'"
        
        cursor.execute(query)
        connection.commit()
        
       


    
    sql_select_Query="SELECT *,STR_TO_DATE(Dated,'%m/%d/%Y') as dat FROM `tags` WHERE MovieID in (SELECT MovieID from movies where Title LIKE '%"+srh+"%')order by dat ASC"
    #print("hello")
    #print(sql_select_Query)
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    
    #searchuserlist= cursor.fetchall
    #print(searchuserlist)
    data = cursor.fetchall()
    
    pdata.clear()

    for i in range(len(data)):
        searchuserlist=[]
        searchuserlist.append(data[i][0])
        searchuserlist.append(data[i][1])
        searchuserlist.append(data[i][2])
        searchuserlist.append(data[i][5])
        searchuserlist.append(data[i][6])
        #searchuserlist.append(data[i][5])
        pdata.append(searchuserlist)
    print(pdata)
    resp = make_response(json.dumps(pdata))
    return resp
    
    #return render_template('search.html',data=pdata)

@app.route('/viewdata')
def view():
    connection = mysql.connector.connect(host='localhost',database='movielensdb',user='root',password='')
    sql_select_Query = "select * from movies"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


    
    return render_template('planning.html', data=data)
    #return render_template('planning.html')
   '''
