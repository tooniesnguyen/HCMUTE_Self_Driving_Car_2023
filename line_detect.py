import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from line import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = build_unet().cuda()
model = build_unet()
model = model.to(device)

model.load_state_dict(torch.load('28.pth', map_location=device))
model.eval()




def process_img(img):
    # 159, 396
 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 80))

    # x = torch.from_numpy(img).cuda()
    x = torch.from_numpy(img).to(device)
    x = x.transpose(1, 2).transpose(0, 1)
    x = x / 255.0
    x = x.unsqueeze(0).float()
    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = pred[0].squeeze()
        pred = (pred > 0.5).cpu().numpy()

        pred = np.array(pred, dtype=np.uint8)
        pred = pred * 255



        kernel = np.ones((5, 5), np.uint8)
        
        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        pred = cv2.erode(pred, kernel, iterations=1)
        pred = cv2.dilate(pred, kernel, iterations=1)



    
    return pred


# pred = process_img(img)

# # cv2.imwrite("pred.png", cv2.resize(pred, (160, 80)))

# plt.imshow(pred)
# plt.show()
