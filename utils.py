'''
Description: 
Author: YaoXianMa
Date: 2022-05-16 13:09:00
LastEditors: YaoXianMa
LastEditTime: 2022-05-16 15:03:35
'''
import cv2
import server_connect
from bluetooth import *
import sys
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def detect_imageLine(image,Lline,Rline):
    cv2.line(image, (int(Lline), 140), (int(Lline), 290), (0, 0, 255), 3) #left
    cv2.line(image, (int(Rline), 140), (int(Rline), 290), (0, 0, 255), 3) #right
    return image


def end_record(vidout, output_file, status,machine_id,t_left_times,t_right_times,durations,reason):
    vidout.release()
    try:
        server_connect.upload2ftp(output_file)
    except:
        pass
    try:
        server_connect.upload2ftp(output_file[0:-3]+'txt')
        server_connect.deletefile(output_file[0:-4]+'txt')
    except:
        pass
    try:
        server_connect.upload2ftp(output_file[0:-4]+'F.jpg')
        server_connect.deletefile(output_file[0:-4]+'F.jpg')
    except:
        pass
    try:
        server_connect.upload2ftp(output_file[0:-4]+'S.jpg')
        server_connect.deletefile(output_file[0:-4]+'S.jpg')
    except:
        pass
    try:
        server_connect.upload2ftp(output_file[0:-4]+'C.jpg')
        server_connect.deletefile(output_file[0:-4]+'C.jpg')
    except:
        pass
    try:
        server_connect.insertSQL(output_file, status,machine_id,t_left_times,t_right_times,durations,reason)
    except:
        pass


def bluetooth(addr, buf_size):
    service_matches = find_service( address = addr )

    if len(service_matches) == 0:
        print("couldn't find the SampleServer service =(")
        sys.exit(0)

    for s in range(len(service_matches)):
        print("\nservice_matches: [" + str(s) + "]:")
        print(service_matches[s])
   
    first_match = service_matches[0]
    port = first_match["port"]
    name = first_match["name"]
    host = first_match["host"]
    port=1
    print("connecting to \"%s\" on %s, port %s" % (name, host, port))

    # Create the client socket
    sock=BluetoothSocket(RFCOMM)
    sock.connect((host, port))
    print("connected")
    return sock
