import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
from model import build_unet
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from collections import Counter
from collections import deque

model_yolo_name='best.pt'
model_yolo = YOLO(model_yolo_name)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Tiếp tục thực hiện chương trình của bạn ở đây
# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                
  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 
global sendBack_angle, sendBack_Speed
angle=0
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
error_arr = np.zeros(5)
modes = deque(maxlen=20)
# error_arr = torch.zeros(5)

pre_t = time.time()
pre_error = 0
flag = 1
D_error = 0
MAX_SPEED = 50.0
SPEED_BRAKE = 45.0
SAFE_SPEED = 40.0
Ratio = 0.1
speed=0

model=YOLO()
""" Load the checkpoint """
checkpoint_path = checkpoint_path = "output_20.pth"
model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed
    print(speed)

def PID(error, p, i, d): #0.43,0,0.02
    global pre_t
    # global error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - pre_t
    # print('DELAY: {:.6f}s'.format(delta_t))
    pre_t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    if abs(angle)>25:
        angle = np.sign(angle)*25
    return int(angle)

def brake(Speed, Speed_limit, ratio):
    brake = ratio*pow(-Speed + Speed_limit, 3)
    if brake > 0:
        return 0
    elif brake < -150:
        return -150
    else:
        return brake

def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    # plt.imshow( image_remove,cmap="gray")
    # plt.show()
    return image_remove

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
            """
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            #print(data)
            try:
                data_recv = json.loads(data)
            except:
                continue

            # Angle and speed recv from server
            # current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            print("speed: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            # -------------------------------------------Workspace---------------------------------- #
            start = time.time()
            image_crop = image[125:, :]
            imageSeg = cv2.resize(image_crop, (160, 80))

            results = model_yolo(image)
            # In ra thông tin của box đầu tiên để khám phá cấu trúc của nó
            # In ra thông tin của box đầu tiên trong list để xác định cấu trúc của nó
            mode = results[0].boxes.cls.cpu().detach().numpy()
            if mode.size > 0:  # Kiểm tra xem mảng 'mode' có phần tử nào không
                mode_int = mode[0].astype(int)  # Chuyển mảng 'mode' sang kiểu int
                modes.append(mode_int)  # Thêm 'mode_int' vào danh sách 'modes'
                modes.append(mode_int)
            print(modes)
            # for result in results:
            #     print(result)


            # for box in results.boxes:
            #     print(f"Class ID: {box.class_id} - Confidence: {box.confidence}")
            # plt.imshow(imageSeg,cmap="gray")
            # plt.show()
            # print("Results", results)
            # print("Result", results[0].boxes.xyxy.cpu().detach().numpy())
            # cv2.imshow("Image Segmentation", image)
            """Detect lane"""
            x = torch.from_numpy(imageSeg)
            x = x.to(device)
            x = x.transpose(1, 2).transpose(0, 1)
            x = x / 255.0
            
            x = x.unsqueeze(0).float()
            with torch.no_grad():
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)

                pred_y = pred_y[0]
                pred_y = pred_y.squeeze()
                pred_y = pred_y > 0.5
                # pred_y = torch.tensor(pred_y, dtype=torch.uint8)
                #print(pred_y[60,0:160])

                pred_y = pred_y.cpu().numpy()
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = pred_y * 255
                mask = pred_y
            # plt.imshow(mask,cmap="gray")
            # plt.show()
            # cv2.imshow("abc", mask)
            pred_y = remove_small_contours(pred_y)
            # """PID controller"""
            arr = []
            arr2 = []
            lineRow = pred_y[50,:]
            if len(lineRow) == 0:
                continue  # Nếu lineRow rỗng, chạy lại vòng lặp
            for x, y in enumerate(lineRow):
                if y == 255:
                    arr = np.append(arr, x)
            # print(arr)
            Min = min(arr)
            Max = max(arr)
            lineRow2 = pred_y[50,:]
            for x, y in enumerate(lineRow2):
                if y == 255:
                    arr2 = np.append(arr2, x)
            Min2 = min(arr2)
            Max2 = max(arr2)
            center = int((Min + Max) / 2)
            error = int(pred_y.shape[1] / 2) - center    ##trai duong, phai ammm
            
            # ------------------TEST----------------#
            if (Min2 == 0) & (Max2 == 159):
                # sendBack_angle = -PID(-30, 0.4, 0.003, 0.05 )
                # sendBack_Speed = 50
                counter = Counter(modes)
                most_common_value = counter.most_common(1)[0][0]
                print("Gia triiiiiiiiiiii nhan duoc la ____________", most_common_value)
                if (most_common_value == 0) or (most_common_value == 3):
                    sendBack_angle = -PID(-30, 0.4, 0.003, 0.05 )
                    sendBack_Speed = 50
                    print("Right....................................................")
                elif (most_common_value == 1) or (most_common_value == 4):
                    sendBack_angle = -PID(30, 0.4, 0.003, 0.05 )
                    sendBack_Speed = 50
                    print("Left....................................................")
                elif (most_common_value == 2):
                    sendBack_angle = -PID(error, 0.4, 0.003, 0.05 )
                    sendBack_Speed = 150
            else:
                sendBack_angle = -PID(error, 0.4, 0.003, 0.05 )
                sendBack_Speed = 150
                if float(current_speed) < 15.0:
                    sendBack_Speed = 150
                elif float(current_speed) > MAX_SPEED:  # Chỉnh Speed (Speed an toàn: 52)
                    sendBack_Speed = 5
                


            Control(sendBack_angle, sendBack_Speed)
            # end = time.time()
            # fps = 1 / (end - start)
            # print(fps)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #      break
            # image = pred_y[20, :]
            # if len(image.shape) == 1:
            #     # Nếu image là mảng 1D, chuyển đổi nó thành mảng 2D
            #     image_2d = np.reshape(image, (1, image.shape[0]))  # Sử dụng (1, image.shape[0]) để tạo ma trận hàng duy nhất
            # else:
            #     image_2d = image

            # plt.imshow(image_2d, cmap="gray")
            # plt.show()

            # print(sendBack_angle)
            # print(sendBack_Speed)

    finally:
        print('closing socket')
        s.close()

