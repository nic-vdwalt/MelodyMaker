import report as r

##Class to get menu input

def menu():
    print("What would you like to do")
    print("1. Generate music")
    print("2. Get help")
    print("3. Show report")
    print("4. Exit")
    
    menuIn = int(input(""))
    if menuIn == 1:
        import rbm_midi_generator
    elif menuIn == 2:
        import help_page
        
    elif menuIn == 3:
        ##add report stuff
        r.getData();
    elif menuIn == 4:
        quit()
    else:
        print("Not a valid option")
        menu()
menu()   
    
    





