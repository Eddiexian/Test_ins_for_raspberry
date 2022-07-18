
import cv2
from cv2 import COLOR_BGR2GRAY
from cv2 import VideoCapture
from cv2 import COLOR_RGBA2BGR
import numpy as np
import time
import math
import queue
import random




def detect_S_move(textfile):
    textfile = open(textfile, "r")

    point_idx=[]
    point_x_list=[]
    point_y_list=[]
    for line in textfile.readlines():
        point_x_list.append(line.split(",")[0])
        point_y_list.append(line.split(",")[1])
        point_idx.append(line.split(",")[2][:-1])
    


        pre_c_x = 0
        pre_c_y = 0
        cnt = 0

        pre_trend = 0

        y_change = queue.Queue(maxsize = 3)
        x_change = queue.Queue(maxsize = 2)

        U=0
        D=0
        R=0
        L=0

        y_status_now = "N"
        x_status_now = "N"

        pre_x_status = "N"
        pre_y_status = "N"

        OK_time = 0
        right_left_cnt = 0
        color = (0,255,0)







        for i in range(len(point_x_list)):
            c_x = int(point_x_list[i])
            c_y = int(point_y_list[i])
            cnt +=1

            if(pre_c_x != 0 and pre_c_y !=0 ):
                
                if(c_x - pre_c_x > 0):
                    x_change.put("R")
                    
                elif(c_x - pre_c_x < 0):
                    x_change.put("L")
                    
                else:
                    x_change.put("N")
                    
                if(c_y - pre_c_y >= 1):
                    y_change.put("D")
                    
                elif(c_y - pre_c_y <= -1):
                    y_change.put("U")
                    
                else:
                    y_change.put("N")
                    

            
                if(y_change.full()):
                    U=0
                    D=0
                    # while(not y_change.empty()):
                    #     status = y_change.get()
                    #     if(status =="U"):
                    #         U+=1
                    #     elif(status=="D"):
                    #         D+=1
                    for i in range (3):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                        status = y_change.get()
                        if(status =="U"):
                            U+=1
                        elif(status=="D"):
                            D+=1
                        if(i!=0):
                            y_change.put(status)
                
                if(U>=2):
                    
                    y_status_now = "U"
                elif(D>=2):
                
                    y_status_now = "D"
                else:
                
                    y_status_now = "N"     
            
                if(x_change.full()):
                    R=0
                    L=0

                    for i in range (2):    
                        status = x_change.get()
                        if(status =="R"):
                            R+=1
                        elif(status=="L"):
                            L+=1
                        if(i!=0):
                            x_change.put(status)    

                    # while(not x_change.empty()):
                    #     status = x_change.get()
                    #     if(status =="R"):
                    #         R+=1
                    #     elif(status=="L"):
                    #         L+=1
                if(R>=1):
                    
                    x_status_now = "R"    
                elif(L>=1):
                    
                    x_status_now = "L"  
                else:
                    
                    x_status_now = "N"    

            pre_c_x = c_x
            pre_c_y = c_y

            if(y_status_now == pre_y_status == "U"):
                if(pre_x_status != x_status_now and pre_x_status!="N"and x_status_now!="N"):
                    right_left_cnt +=1
                    print("U - RL+1")
            elif(y_status_now == pre_y_status == "D"):
                if(pre_x_status != x_status_now and pre_x_status!="N"and x_status_now!="N"):
                    right_left_cnt +=1
                    print("D - RL+1")
            else:
                if(right_left_cnt>=4):
                    OK_time +=1

                if(right_left_cnt != 0):
                    print("RL:",right_left_cnt)  
                print("reset --------------------------")

                right_left_cnt =0

            pre_y_status = y_status_now
            pre_x_status = x_status_now
        
    return OK_time



print(detect_S_move("0.txt"))