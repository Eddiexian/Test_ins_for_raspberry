'''
Description: 
Author: YaoXianMa
Date: 2022-05-25 13:49:36
LastEditors: YaoXianMa
LastEditTime: 2022-06-01 14:08:47
'''


import cv2
from cv2 import COLOR_BGR2GRAY
from cv2 import VideoCapture
import numpy as np
import time
import math
import queue
import random



def detect_point_summary(textfile_str,ROI):

    tracj_img = np.zeros((480, 640, 3), np.uint8)
    # Video = VideoCapture("6f_OCT200_20220601084501.mp4")
    # ret,new1 = Video.read()

    #ROI = cv2.selectROI('rois',new1,False,False)
    textfile = open(textfile_str, "r")
    print(ROI)
    # while Video.isOpened() :
    #     ret,new1 = Video.read()
    #     #time.sleep(0.5)
    #     cv2.imshow("VIDEO",new1)
    
    #     if cv2.waitKey(27) & 0xFF == ord('q') or ret!=True:
    #             break
    # Video.release()
    # #cv2.destroyAllWindows()

    tmpstr = textfile_str[:-4]
    new = np.zeros((480, 640, 3), np.uint8)

    point_idx=[]
    point_x_list=[]
    point_y_list=[]
    for line in textfile.readlines():
        point_x_list.append(line.split(",")[0])
        point_y_list.append(line.split(",")[1])
        point_idx.append(line.split(",")[2][:-1])
    



    x,y,w,h=ROI
    #top line (x,y) ~ (x+w,y)
    #bot line (x,y+h) ~ (x+w,y+h)
    #left line (x,y) ~ (x,y+h)
    #right line (x+w,y) ~ (x+w,y)

    #w/4 h/4

    width = int(w/4)


    height = int(h/4)


    points=[]
    now_point = (x,y)

    for i in range(0,5):  #16 points
        x = now_point[0]+width*(i)
        for j in range(0,5):

            y = now_point[1]+height*(j)

            new = cv2.circle(new,(x,y),2,(255,255,0),1)
            points.append((x,y))

        


    area_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  #16 area counts
    cv2.rectangle(new,(points[0]),(points[6]),(0,255,0),1)
    cv2.rectangle(new,(points[1]),(points[7]),(0,255,0),1)
    cv2.rectangle(new,(points[2]),(points[8]),(0,255,0),1)
    cv2.rectangle(new,(points[3]),(points[9]),(0,255,0),1)

    cv2.rectangle(new,(points[5]),(points[11]),(0,255,0),1)
    cv2.rectangle(new,(points[6]),(points[12]),(0,255,0),1)
    cv2.rectangle(new,(points[7]),(points[13]),(0,255,0),1)
    cv2.rectangle(new,(points[8]),(points[14]),(0,255,0),1)

    cv2.rectangle(new,(points[10]),(points[16]),(0,255,0),1)
    cv2.rectangle(new,(points[11]),(points[17]),(0,255,0),1)
    cv2.rectangle(new,(points[12]),(points[18]),(0,255,0),1)
    cv2.rectangle(new,(points[13]),(points[19]),(0,255,0),1)

    cv2.rectangle(new,(points[15]),(points[21]),(0,255,0),1)
    cv2.rectangle(new,(points[16]),(points[22]),(0,255,0),1)
    cv2.rectangle(new,(points[17]),(points[23]),(0,255,0),1)
    cv2.rectangle(new,(points[18]),(points[24]),(0,255,0),1)

    pre_c_x =0
    pre_c_y =0

    for i in range(len(point_x_list)):
        c_x = int(point_x_list[i])
        c_y = int(point_y_list[i])
        new = cv2.circle(new, (c_x, c_y), 1, (0,0,255), -1)

        # if(pre_c_x != 0 and pre_c_y !=0 ):
        
        #     cv2.line(new,(pre_c_x,pre_c_y),(c_x,c_y),(0,0,255),1)

        # pre_c_x = c_x
        # pre_c_y = c_y
        detect_point = (c_x,c_y)

        if(detect_point[0] >= points[0][0] and detect_point[0] < points[6][0] and detect_point[1] >= points[0][1] and detect_point[1] < points[6][1]):
            area_list[0] += 1
        elif (detect_point[0] >= points[1][0] and detect_point[0] < points[7][0] and detect_point[1] >= points[1][1] and detect_point[1] < points[7][1]):
            area_list[1] += 1 
        elif (detect_point[0] >= points[2][0] and detect_point[0] < points[8][0] and detect_point[1] >= points[2][1] and detect_point[1] < points[8][1]):
            area_list[2] += 1 
        elif (detect_point[0] >= points[3][0] and detect_point[0] < points[9][0] and detect_point[1] >= points[3][1] and detect_point[1] < points[9][1]):
            area_list[3] += 1 

        elif (detect_point[0] >= points[5][0] and detect_point[0] < points[11][0] and detect_point[1] >= points[5][1] and detect_point[1] < points[11][1]):
            area_list[4] += 1 
        elif (detect_point[0] >= points[6][0] and detect_point[0] < points[12][0] and detect_point[1] >= points[6][1] and detect_point[1] < points[12][1]):
            area_list[5] += 1 
        elif (detect_point[0] >= points[7][0] and detect_point[0] < points[13][0] and detect_point[1] >= points[7][1] and detect_point[1] < points[13][1]):
            area_list[6] += 1 
        elif (detect_point[0] >= points[8][0] and detect_point[0] < points[14][0] and detect_point[1] >= points[8][1] and detect_point[1] < points[14][1]):
            area_list[7] += 1 

        elif (detect_point[0] >= points[10][0] and detect_point[0] < points[16][0] and detect_point[1] >= points[10][1] and detect_point[1] < points[16][1]):
            area_list[8] += 1 
        elif (detect_point[0] >= points[11][0] and detect_point[0] < points[17][0] and detect_point[1] >= points[11][1] and detect_point[1] < points[17][1]):
            area_list[9] += 1 
        elif (detect_point[0] >= points[12][0] and detect_point[0] < points[18][0] and detect_point[1] >= points[12][1] and detect_point[1] < points[18][1]):
            area_list[10] += 1 
        elif (detect_point[0] >= points[13][0] and detect_point[0] < points[19][0] and detect_point[1] >= points[13][1] and detect_point[1] < points[19][1]):
            area_list[11] += 1 
            
        elif (detect_point[0] >= points[15][0] and detect_point[0] < points[21][0] and detect_point[1] >= points[15][1] and detect_point[1] < points[21][1]):
            area_list[12] += 1 
        elif (detect_point[0] >= points[16][0] and detect_point[0] < points[22][0] and detect_point[1] >= points[16][1] and detect_point[1] < points[22][1]):
            area_list[13] += 1 
        elif (detect_point[0] >= points[17][0] and detect_point[0] < points[23][0] and detect_point[1] >= points[17][1] and detect_point[1] < points[23][1]):
            area_list[14] += 1 
        elif (detect_point[0] >= points[18][0] and detect_point[0] < points[24][0] and detect_point[1] >= points[18][1] and detect_point[1] < points[24][1]):
            area_list[15] += 1 


    # gray = new[:,:,2]
    # ret, binary = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)  
    
    # contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  

    # cnt = contours[0]
    # # 寻找凸包并绘制凸包（轮廓）
    # print(cnt)
    # hull = cv2.convexHull(cnt)

    # length = len(hull)
    # for i in range(len(hull)):
    #     cv2.line(new, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (255,0,0), 1)

    cv2.imwrite(tmpstr+'F.jpg',new)
    #cv2.imshow('roi', new)
    nnew = new.copy()




    cv2.putText(nnew, str(area_list[0]), (points[0][0]+int(width/3),points[0][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[1]), (points[1][0]+int(width/3),points[1][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[2]), (points[2][0]+int(width/3),points[2][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[3]), (points[3][0]+int(width/3),points[3][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[4]), (points[5][0]+int(width/3),points[5][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[5]), (points[6][0]+int(width/3),points[6][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[6]), (points[7][0]+int(width/3),points[7][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[7]), (points[8][0]+int(width/3),points[8][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[8]), (points[10][0]+int(width/3),points[10][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[9]), (points[11][0]+int(width/3),points[11][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[10]), (points[12][0]+int(width/3),points[12][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[11]), (points[13][0]+int(width/3),points[13][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[12]), (points[15][0]+int(width/3),points[15][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[13]), (points[16][0]+int(width/3),points[16][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[14]), (points[17][0]+int(width/3),points[17][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[15]), (points[18][0]+int(width/3),points[18][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    #cv2.imshow('cnt', nnew)
    cv2.imwrite(tmpstr+'S.jpg',nnew)
    for i in range(0,len(area_list)): #normalize
        if(max(area_list) ==0):
            
            area_list[i] = int(area_list[i]/100)
        else:
            area_list[i] = int(area_list[i]/max(area_list)*100)



    def rect_paint(area_pos,point_pos):
        if(area_list[area_pos]>=90):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,255),-1)
        elif(area_list[area_pos]>=80):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,235),-1)
        elif(area_list[area_pos]>=70):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,215),-1)
        elif(area_list[area_pos]>=60):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,195),-1)
        elif(area_list[area_pos]>=50):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,175),-1)
        elif(area_list[area_pos]>=40):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,155),-1)
        elif(area_list[area_pos]>=30):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,135),-1)
        elif(area_list[area_pos]>=20):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,115),-1)
        elif(area_list[area_pos]>=10):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,95),-1)
        elif(area_list[area_pos]>=1):
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,75),-1)
        else:
            cv2.rectangle(new,points[point_pos],points[point_pos+6],(0,0,0),-1)

    cv2.putText(nnew, str(area_list[0]), (points[0][0]+int(width/3),points[0][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[1]), (points[1][0]+int(width/3),points[1][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[2]), (points[2][0]+int(width/3),points[2][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[3]), (points[3][0]+int(width/3),points[3][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[4]), (points[5][0]+int(width/3),points[5][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[5]), (points[6][0]+int(width/3),points[6][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[6]), (points[7][0]+int(width/3),points[7][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[7]), (points[8][0]+int(width/3),points[8][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[8]), (points[10][0]+int(width/3),points[10][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[9]), (points[11][0]+int(width/3),points[11][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[10]), (points[12][0]+int(width/3),points[12][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[11]), (points[13][0]+int(width/3),points[13][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[12]), (points[15][0]+int(width/3),points[15][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[13]), (points[16][0]+int(width/3),points[16][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[14]), (points[17][0]+int(width/3),points[17][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(nnew, str(area_list[15]), (points[18][0]+int(width/3),points[18][1]+int(height/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)



    #numbers to color and paint
    rect_paint(0,0)
    rect_paint(1,1)
    rect_paint(2,2)
    rect_paint(3,3)
    rect_paint(4,5)
    rect_paint(5,6)
    rect_paint(6,7)
    rect_paint(7,8)
    rect_paint(8,10)
    rect_paint(9,11)
    rect_paint(10,12)
    rect_paint(11,13)
    rect_paint(12,15)
    rect_paint(13,16)
    rect_paint(14,17)
    rect_paint(15,18)

    #cv2.imshow('nwe',new)
 
    cv2.imwrite(tmpstr+"C.jpg",new)



if __name__ == '__main__':
    ROI = (239, 170, 224, 107)
    textfile_str = "6f_OCT200_20220601084501.txt"
    detect_point_summary(textfile_str,ROI)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



