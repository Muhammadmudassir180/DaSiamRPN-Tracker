import torch
import numpy as np
import cv2

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


load = SiamRPNBIG()
load.load_state_dict(torch.load("path of a model"))
load.eval().cuda()


cap = cv2.VideoCapture("path of a video file")
frame = cap.read()[1]


init_box = cv2.selectROI("frame", frame, False)

x1, y1, w, h = init_box

# x1 = cx - 0.5 * w
# y1 = cy - 0.5 * h

cx = x1 + 0.5 * w
cy = y1 + 0.5 * h

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
# im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(frame, target_pos, target_sz, load)


# cap = cv2.VideoCapture(0)
frame = cap.read()[1]
toc = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Camera not connected"
              )
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, frame)
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    fps = toc/cv2.getTickFrequency()
    fps = str(int(fps))
    cv2.rectangle(frame, (res[0], res[1]), (res[0] +
                  res[2], res[1] + res[3]), (0, 255, 255), 3)
    # cv2.putText(frame, "frame:", (10, 100),
    #             cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    # cv2.putText(frame, fps, (100, 100),
    #             cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('SiamRPN', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
