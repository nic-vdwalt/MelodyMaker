import report as r


##genre array with temp values 
arrGenres = ["Blues", 
"Classic", 
"Dance",
"Pop", 
"Punk", 
"Rap", 
"Rock", 
"Jazz",
"Metal" 

]



##mood array with temp values
arrMoods = ["Sad", 
"Happy", 
"Cheerful",
"Romantic", 
"Aggressive", 
"Intense", 
"Calm", 
"Excited", 
"Humorous" 
]


  


##Class to get genres
def Genres():
    for i, x in enumerate(arrGenres):
            print(i+1,x)
    tempGenre = input("Please Select a Genre: ")
    checkGen = False
    if tempGenre.isdigit():
        for i, x in enumerate(arrGenres):
            if int(tempGenre) == i+1:
                varGenre = x
                checkGen = True             
    else:
         for i, x in enumerate(arrGenres):
             if tempGenre.lower() in [x.lower() for x in arrGenres]:
                 checkGen = True
                 varGenre = x
    if checkGen == False:
        print("Please Select a genre: ")
        varGenre = Genres()
    else:
        return varGenre
import datetime

def Moods():
    for i, x in enumerate(arrMoods):
            print(i+1,x)
    tempMood = input("Please Select a mood: ")
    checkMood = False
    if tempMood.isdigit():
        for i, x in enumerate(arrMoods):
            if int(tempMood) == i+1:
                varMood = x
                checkMood = True             
    else:
        for i, x in enumerate(arrMoods):
            if tempGenre.lower() in [x.lower() for x in arrMoods]:
                checkMood = True
                varMood = x

    if checkMood == False:
        print("Please select a mood: ")
        varMood = Moods()
    else:
        return varMood



            





