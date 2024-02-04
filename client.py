import socket       
import sys      
import time
import cv2
import numpy as np
import json
import base64
import keyboard

# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                


# Count
road = 0
car = 0
straight = 0
right = 48
left = 0
no_right = 0
no_left = 0

  
# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 

count = 0
angle = 10
speed = 100
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
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            # print(data)
            try:
                data_recv = json.loads(data)
            except:
                continue

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("speed: ", current_speed)
            # print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            imgage = cv2.imdecode(jpg_as_np, flags=1)
            cv2.imshow("IMG", imgage)
            # print("Img Shape: ",imgage.shape)
            #save image
            # image_name = "./img/img_{}.jpg".format(count)
            # count += 1
            # cv2.imwrite(image_name, imgage)

            key = cv2.waitKey(1)
            # if key & 0xFF == ord('r'):
            #     img_name = "data_img/Road/road_{}.jpg".format(road)
            #     cv2.imwrite(img_name, imgage)
            #     road += 1
            #     print('road: ',road)

            # if key & 0xFF == ord('c'):
            #     img_name = 'data_img/Sign/car_{}.jpg'.format(car)
            #     cv2.imwrite(img_name, imgage)
            #     car += 1
            #     print('car: ',car)

            if key & 0xFF == ord('1'):
                img_name = "C:/Users/Tinh/OneDrive - hcmute.edu.vn/Desktop/Chung_ket/data_img/Sign/straight/straight_{}.jpg".format(straight)
                cv2.imwrite(img_name, imgage)
                straight += 1
                print('straight: ', straight)
            if key & 0xFF == ord('2'):
                img_name = "C:\Users\Tinh\OneDrive - hcmute.edu.vn\Desktop\Chung_ket\data_img\Sign\right\right_{}.jpg".format(right)
                cv2.imwrite(img_name, imgage)
                right += 1 
                print('right: ',right)
            if key & 0xFF == ord('3'):
                img_name = "C:Users\Tinh\OneDrive - hcmute.edu.vn\Desktop\Chung_ket\data_img\Sign\left\left_{}.jpg".format(left)
                cv2.imwrite(img_name, imgage)
                left += 1
                print('left: ',left)
            if key & 0xFF == ord('4'):
                img_name = "C:\Users\Tinh\OneDrive - hcmute.edu.vn\Desktop\Chung_ket\data_img\Sign\no_right\no_right_{}.jpg".format(no_right)
                cv2.imwrite(img_name, imgage)
                no_right += 1          

                print('No right: ', no_right) 
            if key & 0xFF == ord('5'):
                img_name = "C:\Users\Tinh\OneDrive - hcmute.edu.vn\Desktop\Chung_ket\data_img\Sign\no_left\no_left_{}.jpg".format(no_left)
                cv2.imwrite(img_name, imgage)
                no_left += 1
                print('No left: ',no_left)
            if key & 0xFF == ord('q'):
                break
            

    finally:
        print('closing socket')
        s.close()

