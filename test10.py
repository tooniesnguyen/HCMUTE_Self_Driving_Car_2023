import socket       
import cv2
import numpy as np
import json
import base64
import pandas 
from ultralytics import YOLO


import math
import torch
from torchvision.transforms import ToTensor

import time
import matplotlib.pyplot as plt
from line import *

from line_detect import process_img

from pid import PID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                

# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 



import torch
import os
model_yolo_name='best.pt'



model_yolo = YOLO(model_yolo_name)
model_yolo.conf = 0.6



Reduce_Speed = 0
angle = 0
speed = 160
count = 0
Move_Right, Run_Mode_Right = 0, 0
Move_Left, Run_Mode_Left = 0, 0
Move_Forward, Run_Mode_Forward = 0, 0
After_Move_Forward = 0
No_Detect = 0
Ban_Left = 0
Ban_Right = 0
Avoid_car = 0
def move_right(arrmax):
    speed = 30

    # arrmin = min(arr)
    center=int((arrmax-84))
    if arrmax >= 140:
        angle = 25
    else:
        angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-50))) - 5

    print(angle)

    return angle, speed


def move_left():
    speed = 60
    arr = []
    lineRowBLs = img[40:71,:]
    if img[40:71,40:60]:
        cv2.imshow("Cropped Image", img[40:71,40:60])

    return angle, speed

# def move_forward(arrmin):
#     speed = 200

#     angle = 0
#     print(angle)

#     return angle, speed
def move_forward():
    angle_min = 0
    arr = []
    lineRowBLs = img[40:71,:]
    for i in range(len(lineRowBLs)):
        for x,y in enumerate(lineRowBLs[i]):
            if y == 255:
                arr.append(x)
        arrmax = max(arr)
        arrmin = min(arr)
        center=int((arrmax+arrmin)/2)
        angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-(40+i))))
        if abs(angle )< abs(angle_min):
            angle_min = angle
    return angle_min


def Run_Mode_Avoid():
    angle_max = 0
    arr = []
    lineRowBLs = img[40:63,:]
    for i in range(len(lineRowBLs)):
        for x,y in enumerate(lineRowBLs[i]):
            if y == 255:
                arr.append(x)
        arrmax = max(arr)
        arrmin = min(arr)
        center=int((arrmax+arrmin)/2)
        angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-(40+i))))
        if abs(angle)>abs(angle_max):
            angle_max = angle

    return angle_max

if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]

            NOTE: straigh = 19, no right = 17, no left = 16
            Mai coi lai PID Forward
            """
        while True:
            if After_Move_Forward:
                speed = 100

            elif Reduce_Speed ==  1:
                speed = 30
            
            else:
                speed = 160
            # Send data để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            try:
                data_recv = json.loads(data)
            except:
                continue
            

            # Angle and speed recv from serverS
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            print("angle: ", current_angle)
            print("speed: ", current_speed)
            print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)

            name_img = None
            results = model_yolo(img)

            print("Result", results[0].boxes.xyxy.cpu().detach().numpy())
            for result in results:
                try: 
                    xB = int(result.boxes.xyxy.cpu().detach().numpy()[0][2])
                    xA = int(result.boxes.xyxy.cpu().detach().numpy()[0][0])
                    yB = int(result.boxes.xyxy.cpu().detach().numpy()[0][3])
                    yA = int(result.boxes.xyxy.cpu().detach().numpy()[0][1])
                    name_img = result.boxes.cls.cpu().detach().numpy()[0]
                    area = (xB-xA)*(yB-yA)
                    if name_img == 19 and Move_Left == 0 and Move_Right == 0:
                         Move_Forward = 1
                    if  (name_img == 18 or name_img == 15 or name_img == 17 or name_img == 16 or name_img == 4) and Move_Forward == 0 and area > 300:
                         Reduce_Speed =  1
                    if (name_img == 18 or name_img == 16) and area >= 3000 and Move_Forward == 0 and Move_Left == 0:
                        Move_Right = 1
                    if name_img == 16:
                        Ban_Left = 1
                    if (name_img == 4) and area >= 20000 and Move_Forward ==0 and Move_Left == 0 and Move_Right == 0:
                        Avoid_car = 1
                    if (name_img == 15 or name_img == 17) and area >= 4400 and Move_Right == 0 and Move_Forward == 0:
                        Move_Left = 1
                    if name_img == 17:
                        Ban_Right = 1

                    print('result_detect:', name_img)
                    print('Area of box: ',area)
                    
                except:
                     continue
                cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.imshow("IMG_BF", img)

            
          
            # Cut half of picture
            img = img[200:,:,:]

            img = process_img(img)

            speed = 60
            Kp,Ki,Kd = 0.9, 1.7, 0
            if Avoid_car == 0:
                arr = []
                lineRowBL = img[50,:]
                for x,y in enumerate(lineRowBL):
                    if y == 255:
                        arr.append(x)
                arrmax = max(arr)
                arrmin = min(arr)
                center=int((arrmax+arrmin)/2)
                angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-50)))


                

            if Avoid_car == 1 and name_img != 4:
                No_Detect +=1
                if No_Detect >= 3:
                    Reduce_Speed = 0
                    Avoid_car = 0
                    No_Detect = 0
            elif Avoid_car == 1:
                print('########################################Run Mode Avoid#####################################')
                angle = Run_Mode_Avoid()

            if Move_Right == 1 and name_img != 18:
                No_Detect +=1
                print("No detect signal: ", No_Detect)
                if Ban_Left == 0:                    
                    if No_Detect >=80:
                        No_Detect = 0
                        Move_Right = 0
                        Run_Mode_Right = 0
                        Reduce_Speed = 0
                    elif No_Detect >= 25:
                        angle, speed = move_right(arrmax)
                else:
                    print('Ban left mode')
                    if No_Detect >=80:
                        No_Detect = 0
                        Move_Right = 0
                        Run_Mode_Right = 0
                        Reduce_Speed = 0
                        Ban_Left = 0
                    elif No_Detect >= 35:
                        angle, speed = move_right(arrmax)
                    

            elif Move_Left == 1 and (name_img != 15 or name_img != 17):
                No_Detect +=1
                print("No detect signal: ", No_Detect)
                if Ban_Right == 0:                   
                    if No_Detect >= 110:
                        No_Detect = 0
                        Move_Left = 0
                        Run_Mode_Left = 0
                        Reduce_Speed = 0
                    elif No_Detect >= 70:
                        angle, speed = move_left(arrmin)
                else:
                    print('Ban right mode')
                    if No_Detect >= 50:
                        After_Move_Forward = 0
                        No_Detect = 0
                        Move_Left = 0
                        Run_Mode_Left = 0
                        Reduce_Speed = 0
                        Ban_Right = 0
                    elif No_Detect >= 14:
                        angle, speed = move_left(arrmin)

            elif Move_Forward == 1 and name_img != 19:
                No_Detect +=1
                print("No detect signal: ", No_Detect)                    
                if No_Detect >= 50:
                    After_Move_Forward = 1
                    No_Detect = 0
                    Move_Forward = 0
                    Run_Mode_Forward = 0
                    Reduce_Speed = 0
                elif No_Detect >= 10:
                    angle = move_forward()


            # PID
            print('Distance: ',arrmax-center)
            err = angle - current_angle
            if Move_Left != 1:
                angle =  PID(err, Kp,Ki,Kd,mode) #PID(err, Kp, Ki, Kd)    


            cv2.circle(img,(arrmin,70),5,(0,0,0),5)
            cv2.circle(img,(arrmax,70),5,(0,0,0),5)
            cv2.line(img,(center,50),(int(img.shape[1]/2),img.shape[0]),(0,255,255),(5))

            # print("img shape",img.shape) # (80, 160) 
            sup_img = img[30:65,130:160]
            cv2.imshow("Sup img", sup_img)

            cv2.imshow("IMG", img)

            key = cv2.waitKey(1)
        
            

    finally:
        print('closing socket')
        s.close()

