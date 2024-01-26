import numpy as np
import pyedflib
import os.path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import os


def LoadBDF(bdf_file, name="EXG2", start=None, end=None):
    with pyedflib.EdfReader(bdf_file) as f:
        status_index = f.getSignalLabels().index('Status')
        status_size = f.samples_in_file(status_index)
        status = np.zeros((status_size,))
        f.readsignal(status_index, 0, status_size, status)
        status = status.round().astype('int')
        nz_status = status.nonzero()[0]

        stimuli_start = nz_status[0]
        # stimuli_end = nz_status[-1]

        index = f.getSignalLabels().index(name)
        sample_frequency = f.samplefrequency(index)

        # stimuli_start_seconds = int(round(stimuli_start / sample_frequency))
        start = int(stimuli_start + start*sample_frequency)
        end = int(stimuli_start + end*sample_frequency)

        PhysicalMax = f.getPhysicalMaximum(index)
        PhysicalMin = f.getPhysicalMinimum(index)
        DigitalMax = f.getDigitalMaximum(index)
        DigitalMin = f.getDigitalMinimum(index)

        scale_factor = (PhysicalMax - PhysicalMin) / (DigitalMax - DigitalMin)
        dc = PhysicalMax - scale_factor * DigitalMax

        container = np.zeros((end-start,))
        f.readsignal(index, start, end-start, container)
        container = container * scale_factor + dc

        return container, sample_frequency


path = "/media/dsp520/10tb/pytorch_rppgs/DATASETS/MANHOB_HCI/"
sessions = os.listdir(path)
for session_idx in sessions:
    if session_idx.isdigit():
        session_path = path + session_idx + '/'
        vidname = [each for each in os.listdir(session_path) if each.endswith('.avi')]
        vidname = vidname[0]

        bdf = [each for each in os.listdir(session_path) if each.endswith('.bdf')]

        bdf = bdf[0]
        bdf = os.path.join(session_path, bdf)
        vid = os.path.join(session_path, vidname)
        target_vid = session_path + "vid.avi"

        # target_vid_path = os.path.join(session_path, 'vid.avi')
        # if os.path.exists(target_vid):
        #     print(target_vid)
        #     os.remove(target_vid)

        # read up to 1 min of groundtruth
        container, sample_frequency = LoadBDF(bdf, "EXG2", start=0, end=30)
        gt_path = os.path.join(session_path, "ground_truth.txt")
        if os.path.exists(gt_path):
            os.remove(gt_path)

        ffmpeg_extract_subclip(vid, t1=0, t2=30, targetname=target_vid)

        # with VideoFileClip(vid) as video:
        #     new = video.subclip(5, 35)
        #     new.write_videofile(target_vid, audio=False, codec='mpeg4')

        # else:
        #     print("The file does not exist")

        with open(gt_path, 'w') as f:
            f.write('   '.join(str(a) for a in container))
    else:
        continue

# path = '/media/dsp520/4tb/HR_DL/Pytorch_rppgs/DATASETS/UBFC/subject57/'
# hr = []
# hrtrace = []
# time = []
# f = open(path + 'exe_file.txt', 'r')
# f_read = f.read().split('\n')
#
# for temp in f_read:
#     print(temp)
#     i = list(map(float, temp.split()))
#     time.append(i[0]/1000)
#     hr.append(i[1])
#     hrtrace.append(i[3])
#
# hrtrace = (hrtrace - np.mean(hrtrace)) / np.std(hrtrace)
#
# with open(path + 'ground_truth.txt', 'w') as g:
#     g.write('  '.join(str(a) for a in hrtrace))
#     g.write('\n')
#     g.write('  '.join(str(a) for a in hr))
#     g.write('\n')
#     g.write('  '.join(str(a) for a in time))
#     g.write('\n')

# path = '/media/dsp520/4tb/HR_DL/Pytorch_rppgs/DATASETS/MANHOB_HCI/424/ground_truth.txt'
# f = open(path, 'r')
# f_read = f.read()
# print(len(f_read))
# label = f_read.split()
# # i.replace('\U00002013', '-')
# print(label)
# label = list(map(float, label))
# label = np.array(label).astype('float32')
# f.close()

# label = np.interp(np.arange(0, frame_total + 1),
#                   np.linspace(0, frame_total + 1, num=len(label)),
#                   label)

