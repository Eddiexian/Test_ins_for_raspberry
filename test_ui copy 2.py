'''
Description: 
Author: YaoXianMa
Date: 2022-05-16 13:09:00
LastEditors: YaoXianMa
LastEditTime: 2022-05-16 15:04:02
'''


import PySimpleGUI as sg
import cv2
import numpy as np
import yolofastestv2
import time
import detect_point
import config
from datetime import datetime

from utils import *


def start_ui():
    sg.theme('TanBlue')

    video_path = 0

    init_block =[
        [sg.Radio("Video","RADIO_INIT",default=False,key="-Vid-")],
        [sg.Text('檔案位置', size=(8, 1), auto_size_text=False, justification='right'), 
       sg.InputText(enable_events=True,key="-file-"), sg.Button('瀏覽',key = 'getf')], 
        [sg.Radio("Camera","RADIO_INIT",default=True,key="-Cam-")],
        [sg.Button('OK', size=(10, 1))],
    ]
    window_init = sg.Window("Start",init_block)
    
    while True: 
        event, values = window_init.read() 
        if event == 'getf' and values['-Vid-'] == True: 
            text = sg.popup_get_file('請點選瀏覽鍵或自行填入檔案絕對路徑',title = '檔案',file_types = (("video", "*.mp4"),))
            sg.popup('提示', '是否確認選擇檔案---', text) 
            video_path = text
            window_init['-file-'].update(text)  
        if event == 'OK' or event == sg.WIN_CLOSED:
            
            break

      

    block_rules_setting = [
        [sg.Text("EdgeLine & Rules Setting",text_color='red',font = ('Times',14,'bold'))],
        [sg.T("  "),sg.Text('Left bar',size=(50,1),justification = 'center',pad=(0,0),text_color = 'white',background_color='black',)],
        [sg.Slider((0, 640), 270, 1, orientation='h', size=(50, 15),pad=(0,0),key='-left line-',text_color='black')],
        [sg.T("  "),sg.Text('Right bar',size=(50,1),justification = 'center',pad=(0,0),text_color = 'white',background_color='black')],
        [sg.Slider((0, 640), 450, 1, orientation='h', size=(50, 15),pad=(0,0),key='-right line-',text_color='black')],
        [sg.T("  "),sg.Text('L2R cycle times',size=(50,1),justification = 'center',pad=(0,0),text_color = 'white',background_color='black')],
        [sg.Slider((0, 10), 3, 1, orientation='h', size=(50, 15),pad=(0,0),key='-cycle time-')],
        [sg.Text("Time:",text_color='black'),sg.Input("60",key='-time-',size=(7,1))],
        [sg.Text('_'*63,text_color='black')],
        [sg.Text("Image Setting",text_color='red',font = ('Times',14,'bold'))],
        [sg.T("  "),sg.Text('Rotate angle',size=(50,1),justification = 'center',pad=(0,0),text_color = 'white',background_color='black',)],
        [sg.Slider((0, 360), 0, 1, orientation='h', size=(50, 15),pad=(0,0),key='-rotate angle-')],
        [sg.Text('_'*63,text_color='black')],
        [sg.Text("Machine:",text_color='black'),sg.Input("OCT200",key='-machine_id-',size=(7,1))],
        #[sg.T("  "),sg.Text('LT-x,y',justification='right',size=(7,1),pad=(0,0)),sg.Input("100",size=(3,1),key='-ltx-'),sg.Input("100",size=(3,5),key='-lty-')],
        #[sg.T("  "),sg.Text('RD-x,y',justification='right',size=(7,1),pad=(0,0)),sg.Input("100",size=(3,1),key='-rdx-'),sg.Input("100",size=(3,5),key='-rdy-')],
        [sg.Text("Start(Method) Setting",text_color='red',font = ('Times',14,'bold'))],
        [sg.Radio('Start(Image)', "RADIO1", default=False, key="-IN2-",text_color='black'),sg.Button("Select Red light area",key="-select-",button_color="Black"),sg.Input("371,320,394,343",key="-area-",size=(18,1))],
        [sg.Radio('Start(Sensor)', "RADIO1", default=True,text_color='black'),sg.Input("7C:87:CE:49:27:A2",key='-blueaddr-',size=(18,1))],
   
        
        [sg.Button('Go', size=(10, 1))],
    ]

  

    block_image_display = [
        [sg.Image(filename='', key='-IMAGE-')],
    ]



    layout = [
       [[sg.Col(block_rules_setting,key='-img_setting-'),sg.Col(block_image_display,key='-img_display-')]],
       
    ]
    window = sg.Window('Config Setting', layout)


    print(video_path)
    if(video_path != 0):
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if(not ret):
            sg.popup('Camera error!') 
    window_init.close()

    redlightarea=""

    while True:
        event, values = window.read(timeout=20)
 
        left_x = int(values['-left line-'])
        right_x = int(values['-right line-'])
        if event == 'Go' and values['-IN2-']:
            if(values['-area-']==""):
                sg.popup('Please select roi area')
            else:
                break
        elif event == 'Go' and not values['-IN2-']:
            break
        
        
        if event == sg.WIN_CLOSED:
            break
        
        ret, frame = cap.read()


        if(ret):
        
        #-----------------------------------------------
            eframe = frame.copy()

      
            
            frame = rotate_image(frame,values['-rotate angle-'])

            cv2.line(frame,(left_x,0),(left_x,480),(0,0,255),2)
            cv2.line(frame,(right_x,0),(right_x,480),(0,0,255),2)

            #cv2.rectangle(frame,(410,410),(435,430),(0,255,255),0)
            #cv2.rectangle(frame,(int(values['-ltx-']),int(values['-lty-'])),(int(values['-rdx-']),int(values['-rdy-'])),(0,255,255),0)

           

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()   
            window['-IMAGE-'].update(data=imgbytes)
        else:
            frame = rotate_image(eframe,values['-rotate angle-'])

            cv2.line(frame,(left_x,0),(left_x,480),(0,0,255),2)
            cv2.line(frame,(right_x,0),(right_x,480),(0,0,255),2)

            #cv2.rectangle(frame,(410,410),(435,430),(0,255,255),0)
            #cv2.rectangle(frame,(int(values['-ltx-']),int(values['-lty-'])),(int(values['-rdx-']),int(values['-rdy-'])),(0,255,255),0)
        
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)
        
        if event == '-select-':
            r = cv2.selectROI('SELECT ROI', frame, False, False )
            cv2.destroyAllWindows()
            #cv2.waitKey(0)
            redlightarea = str(r[0]) +","+str(r[1])+","+str(r[0]+r[2])+","+str(r[1]+r[3])


            window["-area-"].update(redlightarea)

        
    window.close()
   
    cap.release()
    return values,video_path


def start_test(output_path="", stime=0,cfg=0,video_path=""):
    import cv2

    idle_check = 0
    start_check = 0
    idle_check_nums = 30
    start_check_nums = 3
    go = False
    textfile=open('0.txt','w')
    t_idx=0
    pre_c_x = 0
    pre_c_y = 0
    if(video_path != 0):
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    is_record = False
    record_list = [2]
    
    model = yolofastestv2.yolo_fast_v2(objThreshold=0.5, confThreshold=0.85, nmsThreshold=0.3)

    area_point = cfg["-area-"]
    area_point = (area_point.split(","))

   
  

    while video.isOpened() :
        
        ret, ori_img = video.read()
        res_img = ori_img
        res_img = rotate_image(res_img,cfg['-rotate angle-'])
        ori_img = res_img.copy()  
        cv2.rectangle(res_img,(0,0),(640,100),(0,0,0),-1)
        cv2.rectangle(res_img,(0,380),(640,480),(0,0,0),-1)
        cv2.rectangle(res_img,(520,0),(640,480),(0,0,0),-1)
        cv2.rectangle(res_img,(0,0),(150,480),(0,0,0),-1)
        outputs = model.detect(res_img)
        ori_img,lst = model.postprocess(ori_img, outputs)
        ori_img = detect_imageLine(ori_img,cfg['-left line-'],cfg['-right line-'])
        cv2.imshow("A",res_img)

        if start_check != start_check_nums and lst: 
            start_check+=1
        elif start_check == start_check_nums and not go:
            print("Start")
            go = True
            stime=time.time()
            

        cv2.rectangle((ori_img), (int(area_point[0]), int(area_point[1])), (int(area_point[2]), int(area_point[3])), (255,0,0), 1, cv2.LINE_AA)

        
  

        if go and lst: # reset idle_check
            idle_check=0
        elif go and not lst:  #idle_check++
            idle_check+=1
      
        if(idle_check >= idle_check_nums): #idle_check
            #red light check
            crop_img = ori_img[int(area_point[1]):int(area_point[3]), int(area_point[0]):int(area_point[2])]

            img_target=cv2.inRange(crop_img,(0,0,237),(167,126,255))
            img_clip=cv2.bitwise_and(crop_img,crop_img,mask=img_target)
            img_clip=cv2.cvtColor(img_clip,cv2.COLOR_BGR2GRAY)
            light_pro = cv2.countNonZero(img_clip)
            light_pro /= (int(area_point[2])-int(area_point[0])) * (int(area_point[3])-int(area_point[1]))
            print(light_pro*100)

            if(light_pro > 0):
                print(light_pro*100)
                idle_check=0
                print("got red light idle_check reset",idle_check)
            if(light_pro ==0):    
                go =False
       
            # --- logitcal judgment ---


        
        if go :
            
            if len(lst) > 0:
                for i in lst:
                    c_x = int((i[1]+i[1]+i[3]) /2)
                    c_y = int((i[2]+i[2]+i[4]) /2)
                    
                    t_idx+=1
                    textfile.write(str(c_x)+","+str(c_y)+","+str(t_idx) + "\n")
                    left_line = i[1]
                    right_line = i[1]+i[3]
                    if left_line <= cfg['-left line-'] and record_list[-1]!=0:
                        record_list.append(0)
                    elif right_line >= cfg['-right line-'] and record_list[-1]!=1:
                        record_list.append(1)
                    pre_c_x = c_x
                    pre_c_y = c_y
            # -------------------------
            # show time and record

            #print('FPS {:1f}'.format(1 / (time.time() - stime)))
            status = ""
            cv2.putText(ori_img, f"time:{round(time.time()-stime)} L:{record_list.count(0)} R:{record_list.count(1)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)    
            if not is_record:
                print("Start...")
                output_file = f'{config.eqp}_{str(cfg["-machine_id-"])}_{datetime.now().strftime("%Y%m%d%H%M%S")}.mp4'
                textfile = open(f'{config.eqp}_{str(cfg["-machine_id-"])}_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt','w')
                vidout = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'), 6.0, (640, 480))
                vidout.write(ori_img)
                is_record = True
            if is_record:
                vidout.write(ori_img)
            
        if (not go and idle_check>=idle_check_nums) or (go and round(time.time()-stime)>=500):
            go=False
            print("End...")
            end_time = time.time()
            time_result = end_time-stime
            print("time:", time_result)
            # ---- logical judgment ----
            if len(record_list) >= cfg['-cycle time-']*2+1 and time_result >= int(cfg['-time-']):
                status = "OK"
            else:
                status = "NG"

            # --------------------------
            if(time_result >= 25 and time_result<500 and len(record_list) >=4):
                t_left_times=record_list.count(0)
                t_right_times =record_list.count(1)
                reason="'"
                
                if(len(record_list) <cfg['-cycle time-']*2+1):
                    reason = reason + "1"
                else:
                   
                    reason = reason + "0"
                   
                                           
                
                if(time_result<int(cfg['-time-'])):
                    reason = reason + "1"
                else:
                  
                    reason = reason + "0"
                    
                reason=reason + "'"
                textfile.close()
                ROI = (239, 170, 224, 107)
                textfile_str = output_file[:-4]+'.txt'
                detect_point.detect_point_summary(textfile_str,ROI)
                end_record(vidout, output_file, status,cfg['-machine_id-'],t_left_times,t_right_times,time_result,reason)
                t_idx=0
                textfile = open('0.txt','w')
              


            else:
                server_connect.deletefile(output_file)
            print(status)
            record_list.clear()
            record_list = [2]
            vidout.release()
            cv2.destroyAllWindows()
            if is_record:
                vidout.release()
            start_check= 0
            idle_check = 0
            is_record = False        
       
        winName = 'Detection Test'
        
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, 640, 480)
        cv2.imshow(winName, ori_img)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        #time.sleep(0.1)
    video.release()
    cv2.destroyAllWindows()




def detect_video(stime=0,cfg=0,video_path=""):
    import cv2
#     addr = "94:3C:C6:0D:CF:AE"
    addr = cfg['-blueaddr-']
    buf_size = 1024
    sock = bluetooth(addr=addr, buf_size=buf_size)

    def start(video, stime=0,cfg=0):
        import cv2
        
        pre_c_x = 0
        pre_c_y = 0

        
        is_record = False
        record_list = [2]
        model = yolofastestv2.yolo_fast_v2(objThreshold=0.5, confThreshold=0.8, nmsThreshold=0.3)
        tracj_img = np.zeros((480, 640, 3), np.uint8)

        while True:
            data = sock.recv(buf_size)
            data_2 = int(data[0])
                             
            ret, ori_img = video.read()
            res_img = ori_img
            res_img = rotate_image(res_img,cfg['-rotate angle-'])
            

            ori_img = res_img.copy()  
            cv2.rectangle(res_img,(0,0),(640,140),(0,0,0),-1)
            cv2.rectangle(res_img,(0,400),(640,480),(0,0,0),-1)
            cv2.rectangle(res_img,(450,0),(480,480),(0,0,0),-1)            
            outputs = model.detect(res_img)
            ori_img,lst = model.postprocess(ori_img, outputs)
            ori_img = detect_imageLine(ori_img,cfg['-left line-'],cfg['right line-'])

#             print(stime)
#             print(round(time.time()-stime))
            # show time
            # cv2.putText(res_img, f"time:{round(time.time()-stime)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
            # --- logitcal judgment ---
            if len(lst) > 0:
                for i in lst:
                    c_x = int((i[1]+i[1]+i[3]) /2)
                    c_y = int((i[2]+i[2]+i[4]) /2)
                    tracj_img = cv2.circle(tracj_img, (c_x, c_y), 1, (0, 0, 255), -1)
                    if(pre_c_x !=0 and pre_c_y != 0):
                        cv2.line(tracj_img,(pre_c_x,pre_c_y),(c_x,c_y),(0,0,255),1)
                    left_line = i[1]
                    right_line = i[1]+i[3]
                    if left_line <= cfg['-left line-'] and record_list[-1]!=0:
                        record_list.append(0)
                    elif right_line >= cfg['-right line'] and record_list[-1]!=1:
                        record_list.append(1)
                    pre_c_x = c_x
                    pre_c_y = c_y
            # -------------------------
            # show time and record
            cv2.putText(ori_img, f"time:{round(time.time()-stime)} L:{record_list.count(0)} R:{record_list.count(1)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            winName = 'Detection Test'
            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(winName, 640, 480)
            #cv2.moveWindow(winName, 0, 0)
            cv2.namedWindow("Track", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Track", 640, 480)
            #if(round(time.time()-stime)>500):
            if(round(time.time()-stime) > 500):
                data_2 = 0
  
            #cv2.imshow(winName, res_img)
            cv2.imshow(winName, ori_img)
            cv2.imshow("Track",tracj_img)
            #print('FPS {:1f}'.format(1 / (time.time() - stime)))
            status = ""
            if True:
                if not is_record:
                    print("Start...")
                    output_file = f'{config.eqp}_{str(cfg["-machine_id-"])}_{datetime.now().strftime("%Y%m%d%H%M%S")}.mp4'
                    #output_file = f'{config.eqp}_{config.eqp_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.mp4'
                    vidout = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'), 6.0, (640, 480))
                    #vidout.write(res_img)
                    vidout.write(ori_img)
                    is_record = True
            if is_record:
                vidout.write(ori_img)
            #if cv2.waitKey(27) & data_2 == 0 & pre_sig == 0:
            if data_2 ==0:                          
                print("End...")
                end_time = time.time()
                time_result = end_time-stime
                print("time:", time_result)
                # ---- logical judgment ----
                if len(record_list) >= 11 and time_result>=60:
                    status = "OK"
                else:
                    status = "NG"
#                 L = np.array(over_left)
#                 R = np.array(over_right)
#                 L_s = (np.where(L[:-1] != L[1:])[0].__len__())/2
#                 R_s = (np.where(R[:-1] != R[1:])[0].__len__())/2
#                 if L_s >= 5 and R_s >= 5:
#                     status = "OK"
#                 else:
#                     status = "NG"
                # --------------------------
                if(time_result >= 25 and time_result<500):
                    end_record(vidout, output_file, status,cfg['-machine_id-'])
                else:
                    server_connect.deletefile(output_file)
                record_list.clear()
                record_list = [2]
                vidout.release()
                cv2.destroyAllWindows()
                if is_record:
                    vidout.release()
                break
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
            pre_sig = data_2
            #print(q.empty())
    def rx_and_echo(video,output_path=""):
        buf_size = 1024
        pre= 0
        
        
        
        while True:
            #time.sleep(1)
            data = sock.recv(buf_size)
            
            data_2 = int(data[0])
            #print(data_2)
           
                              
            #print("S_Q ",start_q.empty())
            if data_2 == 1 :
                start_time = time.time()
                start(video=video,stime=start_time,cfg=cfg)
                continue
            #print(data_2)
    if(video_path != 0):
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_AUTOFOCUS, 0)       
    rx_and_echo(video,output_path="")






cfg,video_path = start_ui()

if(cfg['-IN2-'] == True):
    start_test("",0,cfg,video_path)
else:
    detect_video(0,cfg,video_path)
