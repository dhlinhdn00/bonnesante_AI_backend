import numpy as np


# def LoadBDF(bdf_file, name="EXG2", start=None, end=None):
#     with pyedflib.EdfReader(bdf_file) as f:
#         status_index = f.getSignalLabels().index('Status')
#         sample_frequency = f.samplefrequency(status_index)
#         status_size = f.samples_in_file(status_index)
#         status = np.zeros((status_size,), dtype='float64')
#         f.readsignal(status_index, 0, status_size, status)
#         status = status.round().astype('int')
#         nz_status = status.nonzero()[0]
#
#         video_start = nz_status[0]
#         video_end = nz_status[-1]
#
#         index = f.getSignalLabels().index(name)
#         sample_frequency = f.samplefrequency(index)
#
#         video_start_seconds = video_start / sample_frequency
#
#         if start is not None:
#             start += video_start_seconds
#             start *= sample_frequency
#             if start < video_start:
#                 start = video_start
#             start = int(start)
#         else:
#             start = video_start
#
#         if end is not None:
#             end += video_start_seconds
#             end *= sample_frequency
#             if end > video_end:
#                 end = video_end
#             end = int(end)
#         else:
#             end = video_end
#
#         PhysicalMax = f.getPhysicalMaximum(index)
#         PhysicalMin = f.getPhysicalMinimum(index)
#         DigitalMax = f.getDigitalMaximum(index)
#         DigitalMin = f.getDigitalMinimum(index)
#
#         scale_factor = (PhysicalMax - PhysicalMin) / (DigitalMax - DigitalMin)
#         dc = PhysicalMax - scale_factor * DigitalMax
#
#         container = np.zeros((end - start,), dtype='float64')
#         f.readsignal(index, start, end - start, container)
#         container = container * scale_factor + dc
#
#         return container, sample_frequency

import numpy as np
import os

path = "/media/dsp520/4tb/HR_DL/Pytorch_rppgs/DATASETS/UBFC/"
for i in range(51, 58):
    file_name = 'subject' + str(i) + '/exe_file.txt'
    path_file = os.path.join(path, file_name)
    with open(path_file, "r") as f:
        f_read = f.readlines()
        print(np.shape(f_read))
        time = f_read[0]
        hr = f_read[2]
        trace = f_read[4]
        f.close()

    file = open(path_file, "w")
    file.write(trace + '\n')
    file.write(hr + '\n')
    file.write(time + '\n')
    file.close()

