import cv2
import numpy as np
from tqdm import tqdm
from face_recognition import face_locations, face_landmarks
import mediapipe as mp
import gc


def preprocess_Video_RGB_only(path, flag, vid_res):
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_video = np.empty((frame_total, vid_res, vid_res, 3), dtype=np.uint8)
    j = 0

    with tqdm(total=frame_total, position=0, leave=True, desc=path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                return False, None
            if flag:
                rst, crop_frame = faceDetection(frame)
                if not rst:
                    return False, None

            crop_frame = cv2.resize(crop_frame, dsize=(vid_res, vid_res), interpolation=cv2.INTER_AREA)
            raw_video[j] = crop_frame
            j += 1
            if j == frame_total:
                break
            pbar.update(1)
        cap.release()
    pbar.close()
    del pbar, j, flag, rst, ret, frame, frame_total
    gc.collect()

    if np.isnan(raw_video).any():
        print('Nan value detected')
        del raw_video
        return False, None
    if np.isinf(raw_video).any():
        print('Infinite value detected')
        del raw_video
        return False, None

    return True, raw_video


def faceLandmarks(frame):
    resized_frame = frame
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    face_location = face_locations(resized_frame, model="cnn")
    if len(face_location) == 0:  # can't detect face
        return False, None, None
    face_landmark_list = face_landmarks(resized_frame)
    i = 0
    center_list = []
    for face_landmark in face_landmark_list:
        for facial_feature in face_landmark.keys():
            for center in face_landmark[facial_feature]:
                center_list.append(center)
                i = i+1
    pt = np.array([center_list[2], center_list[3], center_list[31]])
    pt1 = np.array([center_list[13], center_list[14], center_list[35]])
    pt2 = np.array([center_list[6], center_list[7], center_list[65]])
    pt3 = np.array([center_list[9], center_list[10], center_list[61]])
    dst = cv2.fillConvexPoly(grayscale_frame, pt, color=(255,255,255))
    dst = cv2.fillConvexPoly(dst, pt1, color=(255, 255, 255))
    dst = cv2.fillConvexPoly(dst, pt2, color=(255, 255, 255))
    dst = cv2.fillConvexPoly(dst, pt3, color=(255, 255, 255))
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] != 255:
                dst[i][j] = 0
    top, right, bottom, left = face_location[0]
    dst = resized_frame[top:bottom, left:right]
    mask = grayscale_frame[top:bottom, left:right]
    # test = cv2.bitwise_and(dst,dst,mask=mask)

    return True, dst, mask


def faceDetection(frame):
    global locat

    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    face_location = face_locations(resized_frame, model='cnn')

    if len(face_location) == 0:  # cant detect face
        print('cant detect face')
        return False, None
        # if len(locat[0]) != 4:  # 기존 frame
        #     # dst = resized_frame[resized_frame.shape[0] // 4: resized_frame.shape[0] // 4 * 3,
        #     #       resized_frame.shape[1] // 4:resized_frame.shape[1] // 4 * 3]
        # else:
        #     top, right, bottom, left = locat[0]
        #     dst = resized_frame[max(0, top - 10):min(resized_frame.shape[0], bottom + 10),
        #           max(0, left - 10):min(resized_frame.shape[1], right + 10)]
        # return False, dst
        #     return True, dst

    top, right, bottom, left = face_location[0]
    dst = resized_frame[max(0, top - 10):min(resized_frame.shape[0], bottom + 10),
          max(0, left - 10):min(resized_frame.shape[1], right + 10)]
    locat = face_location
    return True, dst


# def generate_Floatimage(frame):
#     '''
#     :param frame: roi frame
#     :return: float value frame [0 ~ 1.0]
#     use: to normalize frame value (0, 255) -> (0, 1)
#     '''
#     dst = img_as_float(frame)
#     # 왜 있지??
#     dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
#     dst[dst > 1] = 1  # clipping overflow value
#     dst[dst < 0] = 0
#     return dst


def generate_MotionDifference(prev_frame, crop_frame):
    # Motion difference (optical flow) generation between 2 frames (not normalized)
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame + 1)
    return motion_input


def normalize_Image(frame):
    # Frame standardization
    frame = frame / 255.
    frame = np.subtract(frame, np.mean(frame, (0, 1), keepdims=True))
    return frame


def preprocess_Image(prev_frame, crop_frame):
    prev_frame = prev_frame.astype(np.float32)
    crop_frame = crop_frame.astype(np.float32)
    return generate_MotionDifference(prev_frame, crop_frame), normalize_Image(prev_frame)


def ci99(motion_diff):
    max99 = np.mean(motion_diff) + (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    min99 = np.mean(motion_diff) - (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    motion_diff[motion_diff > max99] = max99
    motion_diff[motion_diff < min99] = min99
    return motion_diff


def video_normalize(channel):
    channel /= np.std(channel)
    return channel


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(img)
        #self.faces = self.faceDetection.process(img)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #           0.7, (0, 255, 0), 1)

                    # print(id,x,y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def avg(a, b):
    return [int((x + y) / 2) for x, y in zip(a, b)]


def crop_mediapipe(detector, frame):
    _, dot = detector.findFaceMesh(frame)
    if len(dot) > 0:
        x_min = min(np.array(dot[0][:]).T[0])
        y_min = min(np.array(dot[0][:]).T[1])
        x_max = max(np.array(dot[0][:]).T[0])
        y_max = max(np.array(dot[0][:]).T[1])
        x_center = (int)((x_min + x_max) / 2)
        y_center = (int)((y_min + y_max) / 2)
        if (x_max - x_min) > (y_max - y_min):
            w_2 = (int)((x_max - x_min) / 2)  # This w_2 acts as a spacing factor adding more space into the face box
        else:
            w_2 = (int)((y_max - y_min) / 2)
        f = frame[y_center - w_2 - 10:y_center + w_2 + 10, x_center - w_2 - 10:x_center + w_2 + 10]
        _, dot = detector.findFaceMesh(f)  # Refind the face gain
        return f, dot[0]


def make_mask(dot):
    view_mask = []
    view_mask.append(np.array(
        [
            dot[152],dot[377],dot[400],dot[378],dot[379],dot[365],dot[397],
            dot[288],dot[301],dot[352],dot[447],dot[264],dot[389],dot[251],
            dot[284],dot[332],dot[297],dot[338],dot[10],  dot[109],dot[67],
            dot[103],dot[54]  ,dot[21]  ,dot[162],dot[127],dot[234],dot[93],
            dot[132],dot[215],dot[58]  ,dot[172],dot[136],dot[150],dot[149],
            dot[176],dot[148]
        ]
    ))
    remove_mask = []
    remove_mask.append(np.array(
        [
            dot[37],dot[39],dot[40],dot[185],dot[61],dot[57],dot[43],dot[106],dot[182],dot[83],
            dot[18],dot[313],dot[406],dot[335],dot[273],dot[287],dot[409],dot[270],dot[269],
            dot[267],dot[0],dot[37]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[37],dot[0],dot[267],dot[326],dot[2],dot[97],dot[37]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[2],dot[326],dot[327],dot[278],dot[279],dot[360],dot[363],
            dot[281],dot[5],dot[51],dot[134],dot[131],dot[49],dot[48],
            dot[98],dot[97],dot[2]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[236],dot[134],dot[51],dot[5],dot[281],dot[363],dot[456],
            dot[399],dot[412],dot[465],dot[413],dot[285],dot[336],dot[9],
            dot[107],dot[55],dot[189],dot[245],dot[188],dot[174],dot[236]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[336],dot[296],dot[334],dot[293],dot[283],dot[445],dot[342],dot[446],
            dot[261],dot[448],dot[449],dot[450],dot[451],dot[452],dot[453],dot[464],
            dot[413],dot[285],dot[336]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[107],dot[66],dot[105],dot[63],dot[53],dot[225],dot[113],dot[226],
            dot[31],dot[228],dot[229],dot[230],dot[231],dot[232],dot[233],dot[244],
            dot[189],dot[55],dot[107]
        ]
    ))

    return view_mask, remove_mask


def generate_maks(src, view, remove):
    shape = src.shape
    view_mask = np.zeros((shape[0], shape[1], 3), np.uint8)
    for (idx, mask) in enumerate(view):
        view_mask = cv2.fillConvexPoly(view_mask, mask.astype(int), color=(255, 255, 255))
    remove_mask = np.zeros((shape[0], shape[1], 3), np.uint8)
    for (idx,mask) in enumerate(remove):
        remove_mask = cv2.fillConvexPoly(remove_mask, mask.astype(int), color=(255, 255, 255))

    img = cv2.subtract(view_mask, remove_mask)

    rst = cv2.bitwise_and(src, img)
    return rst
