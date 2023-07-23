import cv2
import numpy as np
import face_recognition as fr
import time
from rich.console import Console
from data import recognize, get_users
from PIL import Image

  
def menu():
    print("     __  __                          ")
    print("    |  \/  | ___ _ __  _   _         ")
    print("    | |\/| |/ _ \ '_ \| | | |        ")
    print("    | |  | |  __/ | | | |_| |        ")
    print("----|_|  |_|\___|_| |_|\__,_|--------")
    print("|                                   |")
    print("|   [1] Autentication               |")
    print("|   [2] Registration                |")
    print("|   [3] Testing                     |")
    print("|   [4] Exit                        |")
    print("|                                   |")
    print("-------------------------------------")    
    
    
def testing_menu():
    print("                                     ")
    print("-------------------------------------")
    print("|                                   |")
    print("|   [1] Images                      |")
    print("|   [2] Camera                      |")
    print("|                                   |")
    print("-------------------------------------")     


console = Console()

with console.status("[bold green]Loading...") as status:
    all_useres= []
    users_name_database = []

    # user1 = recognize("./img/diogo_mestre.jpeg")

    # if(user1[0]):
    #     all_useres.append(user1[1][0])
    #     users_name_database.append("Diogo Mestre")
        
    user2 = recognize("./img/rodrigo_alves.png")

    if(user2[0]):
        all_useres.append(user2[1][0])
        users_name_database.append("Rodrigo Alves")
        
    print("########################################################################################")
            
    print(" _____                  ____                                   _  _    _               ")  
    print("|  ___|__ _   ___  ___ |  _ \   ___   ___  ___    __ _  _ __  (_)| |_ (_)  ___   _ __  ")
    print("| |_  / _` | / __|/ _ \| |_) | / _ \ / __|/ _ \  / _` || '_ \ | || __|| | / _ \ | '_ \ ")
    print("|  _|| (_| || (__|  __/|  _ < |  __/| (__| (_) || (_| || | | || || |_ | || (_) || | | |")
    print("|_|   \__,_| \___|\___||_| \_\ \___| \___|\___/  \__, ||_| |_||_| \__||_| \___/ |_| |_|")
    print("                                                 |___/                                 ")
    print("")
    print("########################################################################################")



while True:
    menu()

    option = int(input("Choose an option:"))
    
    if (option == 1):
        #known_users, name_known_users = get_users()

        capture = cv2.VideoCapture(0)

        while True:
            loop = True
            ret, frame = capture.read()
            
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            
            user_locations = fr.face_locations(rgb_frame)
            unknown_face_encondings = fr.face_encodings(rgb_frame, user_locations)
            
            for (top, right, bottom, left), face_enconding in zip(user_locations, unknown_face_encondings):
                
                result = fr.compare_faces(all_useres, face_enconding)
                #print(result)
                
                face_distance = fr.face_distance(all_useres, face_enconding)
                
                best_match_index = np.argmin(face_distance)
                if (result[best_match_index]):
                    name = users_name_database[best_match_index]
                    loop = False
                else:
                    name = "Unknown"
                    
                cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255),2)
                
                cv2.rectangle(frame, (left,bottom -35), (right,bottom), (0,0,255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)
                
                cv2.imshow('Webcam_facerecognition', frame)
                #time.sleep(5)
                   
                if (not loop):
                    break
                
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
            if (not loop):
                print ("Welcome " + name)
                break
          
        capture.release()
        cv2.destroyAllWindows()
        
    elif (option == 2):
        while True:
            new_name = input("Name: ")
            new_image_path = input("Image Path: ")
            
            image=Image.open(new_image_path)
            image.show()
            
            insert = input("Create a new login? (y/n): ")
            if insert=="y":
                new_user = recognize(new_image_path)
                
                if(new_user[0]):
                    all_useres.append(new_user[1][0])
                    users_name_database.append(new_name)
                    print("User Created Success!")
                    break
            else:
                pass
        
    elif (option == 3):
        testing_menu()
        
        option2 = int(input("Choose an option:"))
        
        if (option2 == 1):
            new_image_path_test = input("Image Path: ")
            
            unknown = recognize(new_image_path_test)
            if(unknown[0]):
                user_unknown = unknown[1][0]
                #known_users, name_known_users = get_users()
                
                result = fr.compare_faces(all_useres, user_unknown)
                print(result)
                
                for i in range(len(all_useres)):
                    results = result[i]
                    
                    if(results):
                        print("User names: " + users_name_database[i], "has been recognized")
            else:
                print("User face not found")
                
        elif (option2 == 2):
            
            capture = cv2.VideoCapture(0)

            while True:
                ret, frame = capture.read()
                
                rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
                
                user_locations = fr.face_locations(rgb_frame)
                unknown_face_encondings = fr.face_encodings(rgb_frame, user_locations)
                
                for (top, right, bottom, left), face_enconding in zip(user_locations, unknown_face_encondings):
                    
                    result = fr.compare_faces(all_useres, face_enconding)
                    # print(result)
                    
                    face_distance = fr.face_distance(all_useres, face_enconding)
                    
                    best_match_index = np.argmin(face_distance)
                    if (result[best_match_index]):
                        name = users_name_database[best_match_index]
                    else:
                        name = "Unknown"
                        
                    cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255),2)
                    
                    cv2.rectangle(frame, (left,bottom -35), (right,bottom), (0,0,255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)
                    
                    cv2.imshow('Webcam_facerecognition', frame)
                    
                    
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            
            capture.release()
            cv2.destroyAllWindows()
            
        else:
            print("Invalid option.\n")
            
    elif (option == 4):
        break
    else:
        print("Invalid option: Try again.\n")
    