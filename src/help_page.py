def displayQuestion():
        print("1. How to generate Music")
        print("2. selecting a genre")
        print("3. Selecting a mood")
        print("4. Back")
        selection = int(input("choose option:"))
        if selection==1:
            print("Step 1: select a genre by entering the number of the genre you wish to use, \nStep 2: select a mood by entering the number of the mood you wish to use, Step \n3: After step 1 and 2 is completed the application will genrate your song")
        elif selection==2:
            print("Choose the number next to the genre to generate that type of genre")
        elif selection==3:
            print("Choose the number next to the mood to generate that type of mood")
        elif selection==4:
            print("")
        else:
            print("Invalid choice. Enter 1-4")
            
            
displayQuestion()