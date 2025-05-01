import datetime
import pickle
import os
import pandas as pd

##class for data
class ReportData():
    genre = ""
    mood = ""
    date = datetime
    


def updateData(genre, mood):
   ##populate rd
    rd = ReportData()
    rd.genre = genre
    rd.mood = mood
    rd.date = datetime.datetime.now()

    
    ##
    exists = os.path.isfile("data.pkl")
    lst = []
    if exists:
        with open("data.pkl", "rb") as pfile:

            
            while 1:
                try:        
                    lst.append(pickle.load(pfile))
                except EOFError:
                    break
            lst.append(rd)
        with open("data.pkl", "wb") as pfile:
            for obj in lst:
                pickle.dump(obj, pfile)
    else:
        lst.append(rd)
        with open("data.pkl", "wb") as pfile:
            for obj in lst:
                pickle.dump(obj, pfile)

def getData():
    ##GET INFORMATION FROM FILE
    exists = os.path.isfile("data.pkl")
    lst = []
    ##populate list with data information
    if exists:
        with open("data.pkl", "rb") as pfile:
            while 1:
                try:
                    
                    lst.append(pickle.load(pfile))
                except EOFError:
                    break
       
        ##add data to data frame
        gs =[]
        ms=[]
        ds=[]
        i = 0
        for l in lst:
            ++i
            gs.append(l.genre)
            ms.append(l.mood)
            ds.append(l.date)
            
            
        
        lst2 = {'Genre': gs,
                'Mood': ms,
                'Date':ds }
        ##print data
        print("Total:"+ str(i))
        print("===================")
        print("Genres")
        print("===================")
        df = pd.DataFrame.from_dict(lst2) 
        print(df.groupby('Genre')['Genre'].count().reset_index(name='Amount'))
        
        print("===================")  
        print("Moods")
        print("===================")   
        print(df.groupby('Mood')['Genre'].count().reset_index(name='Amount'))
        
        print("===================")  
        print("All")
        print("===================")   
        print(df.sort_values(by='Date', ascending = False))
    else:
        print("No data")
    
    


