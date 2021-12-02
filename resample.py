import os
import csv
import numpy as np
from scipy import signal
# import neurokit2 as nk
# import matplotlib.pyplot as plt

stimuli = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4', 'stimuli_5', 'stimuli_6', 'stimuli_7',
           'stimuli_8', 'stimuli_9', 'stimuli_10', 'stimuli_11', 'stimuli_12', 'stimuli_13']

# # resample ECG signals
#
# path = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Participant_Receiver/Physio/'
# folders = os.listdir(path)
# for i in folders:
#     if i != '.DS_Store':
#         print(i)
#         os.chdir(path + i)
#         for j in stimuli:
#             with open(j + '.csv') as in_file, open(j + '_resampled.csv', 'w') as out_file:
#                 file_content = csv.reader(in_file, delimiter=',')
#                 headers = next(file_content, None) # skip headers
#
#                 ecg1_raw = []
#                 ecg2_raw = []
#                 gsr1_raw = []
#                 gsr2_raw = []
#                 erg1_raw = []
#                 erg2_raw = []
#                 resp_raw = []
#                 plet_raw = []
#                 temp_raw = []
#
#                 for row in file_content:
#                     ecg1_raw.append(float(row[0]))
#                     ecg2_raw.append(float(row[1]))
#                     gsr1_raw.append(float(row[2]))
#                     gsr2_raw.append(float(row[3]))
#                     erg1_raw.append(float(row[4]))
#                     erg2_raw.append(float(row[5]))
#                     resp_raw.append(float(row[6]))
#                     plet_raw.append(float(row[7]))
#                     temp_raw.append(float(row[8]))
#
#                     ecg1_new = signal.decimate(ecg1_raw, 13)
#                     ecg2_new = signal.decimate(ecg2_raw, 13)
#                     gsr1_new = signal.decimate(gsr1_raw, 13)
#                     gsr2_new = signal.decimate(gsr2_raw, 13)
#                     erg1_new = signal.decimate(erg1_raw, 13)
#                     erg2_new = signal.decimate(erg2_raw, 13)
#                     resp_new = signal.decimate(resp_raw, 13)
#                     plet_new = signal.decimate(plet_raw, 13)
#                     temp_new = signal.decimate(temp_raw, 13)
#
#                 if j == 'stimuli_1':
#                     ecg1_new = signal.resample(ecg1_new, 2578)
#                     ecg2_new = signal.resample(ecg2_new, 2578)
#                     gsr1_new = signal.resample(gsr1_new, 2578)
#                     gsr2_new = signal.resample(gsr2_new, 2578)
#                     erg1_new = signal.resample(erg1_new, 2578)
#                     erg2_new = signal.resample(erg2_new, 2578)
#                     resp_new = signal.resample(resp_new, 2578)
#                     plet_new = signal.resample(plet_new, 2578)
#                     temp_new = signal.resample(temp_new, 2578)
#                 elif j == 'stimuli_2':
#                     ecg1_new = signal.resample(ecg1_new, 4010)
#                     ecg2_new = signal.resample(ecg2_new, 4010)
#                     gsr1_new = signal.resample(gsr1_new, 4010)
#                     gsr2_new = signal.resample(gsr2_new, 4010)
#                     erg1_new = signal.resample(erg1_new, 4010)
#                     erg2_new = signal.resample(erg2_new, 4010)
#                     resp_new = signal.resample(resp_new, 4010)
#                     plet_new = signal.resample(plet_new, 4010)
#                     temp_new = signal.resample(temp_new, 4010)
#                 elif j == 'stimuli_3':
#                     ecg1_new = signal.resample(ecg1_new, 3921)
#                     ecg2_new = signal.resample(ecg2_new, 3921)
#                     gsr1_new = signal.resample(gsr1_new, 3921)
#                     gsr2_new = signal.resample(gsr2_new, 3921)
#                     erg1_new = signal.resample(erg1_new, 3921)
#                     erg2_new = signal.resample(erg2_new, 3921)
#                     resp_new = signal.resample(resp_new, 3921)
#                     plet_new = signal.resample(plet_new, 3921)
#                     temp_new = signal.resample(temp_new, 3921)
#                 elif j == 'stimuli_4':
#                     ecg1_new = signal.resample(ecg1_new, 2989)
#                     ecg2_new = signal.resample(ecg2_new, 2989)
#                     gsr1_new = signal.resample(gsr1_new, 2989)
#                     gsr2_new = signal.resample(gsr2_new, 2989)
#                     erg1_new = signal.resample(erg1_new, 2989)
#                     erg2_new = signal.resample(erg2_new, 2989)
#                     resp_new = signal.resample(resp_new, 2989)
#                     plet_new = signal.resample(plet_new, 2989)
#                     temp_new = signal.resample(temp_new, 2989)
#                 elif j == 'stimuli_5':
#                     ecg1_new = signal.resample(ecg1_new, 3555)
#                     ecg2_new = signal.resample(ecg2_new, 3555)
#                     gsr1_new = signal.resample(gsr1_new, 3555)
#                     gsr2_new = signal.resample(gsr2_new, 3555)
#                     erg1_new = signal.resample(erg1_new, 3555)
#                     erg2_new = signal.resample(erg2_new, 3555)
#                     resp_new = signal.resample(resp_new, 3555)
#                     plet_new = signal.resample(plet_new, 3555)
#                     temp_new = signal.resample(temp_new, 3555)
#                 elif j == 'stimuli_6':
#                     ecg1_new = signal.resample(ecg1_new, 3277)
#                     ecg2_new = signal.resample(ecg2_new, 3277)
#                     gsr1_new = signal.resample(gsr1_new, 3277)
#                     gsr2_new = signal.resample(gsr2_new, 3277)
#                     erg1_new = signal.resample(erg1_new, 3277)
#                     erg2_new = signal.resample(erg2_new, 3277)
#                     resp_new = signal.resample(resp_new, 3277)
#                     plet_new = signal.resample(plet_new, 3277)
#                     temp_new = signal.resample(temp_new, 3277)
#                 elif j == 'stimuli_7':
#                     ecg1_new = signal.resample(ecg1_new, 3762)
#                     ecg2_new = signal.resample(ecg2_new, 3762)
#                     gsr1_new = signal.resample(gsr1_new, 3762)
#                     gsr2_new = signal.resample(gsr2_new, 3762)
#                     erg1_new = signal.resample(erg1_new, 3762)
#                     erg2_new = signal.resample(erg2_new, 3762)
#                     resp_new = signal.resample(resp_new, 3762)
#                     plet_new = signal.resample(plet_new, 3762)
#                     temp_new = signal.resample(temp_new, 3762)
#                 elif j == 'stimuli_8':
#                     ecg1_new = signal.resample(ecg1_new, 3672)
#                     ecg2_new = signal.resample(ecg2_new, 3672)
#                     gsr1_new = signal.resample(gsr1_new, 3672)
#                     gsr2_new = signal.resample(gsr2_new, 3672)
#                     erg1_new = signal.resample(erg1_new, 3672)
#                     erg2_new = signal.resample(erg2_new, 3672)
#                     resp_new = signal.resample(resp_new, 3672)
#                     plet_new = signal.resample(plet_new, 3672)
#                     temp_new = signal.resample(temp_new, 3672)
#                 elif j == 'stimuli_9':
#                     ecg1_new = signal.resample(ecg1_new, 3701)
#                     ecg2_new = signal.resample(ecg2_new, 3701)
#                     gsr1_new = signal.resample(gsr1_new, 3701)
#                     gsr2_new = signal.resample(gsr2_new, 3701)
#                     erg1_new = signal.resample(erg1_new, 3701)
#                     erg2_new = signal.resample(erg2_new, 3701)
#                     resp_new = signal.resample(resp_new, 3701)
#                     plet_new = signal.resample(plet_new, 3701)
#                     temp_new = signal.resample(temp_new, 3701)
#                 elif j == 'stimuli_10':
#                     ecg1_new = signal.resample(ecg1_new, 3178)
#                     ecg2_new = signal.resample(ecg2_new, 3178)
#                     gsr1_new = signal.resample(gsr1_new, 3178)
#                     gsr2_new = signal.resample(gsr2_new, 3178)
#                     erg1_new = signal.resample(erg1_new, 3178)
#                     erg2_new = signal.resample(erg2_new, 3178)
#                     resp_new = signal.resample(resp_new, 3178)
#                     plet_new = signal.resample(plet_new, 3178)
#                     temp_new = signal.resample(temp_new, 3178)
#                 elif j == 'stimuli_11':
#                     ecg1_new = signal.resample(ecg1_new, 3301)
#                     ecg2_new = signal.resample(ecg2_new, 3301)
#                     gsr1_new = signal.resample(gsr1_new, 3301)
#                     gsr2_new = signal.resample(gsr2_new, 3301)
#                     erg1_new = signal.resample(erg1_new, 3301)
#                     erg2_new = signal.resample(erg2_new, 3301)
#                     resp_new = signal.resample(resp_new, 3301)
#                     plet_new = signal.resample(plet_new, 3301)
#                     temp_new = signal.resample(temp_new, 3301)
#                 elif j == 'stimuli_12':
#                     ecg1_new = signal.resample(ecg1_new, 3265)
#                     ecg2_new = signal.resample(ecg2_new, 3265)
#                     gsr1_new = signal.resample(gsr1_new, 3265)
#                     gsr2_new = signal.resample(gsr2_new, 3265)
#                     erg1_new = signal.resample(erg1_new, 3265)
#                     erg2_new = signal.resample(erg2_new, 3265)
#                     resp_new = signal.resample(resp_new, 3265)
#                     plet_new = signal.resample(plet_new, 3265)
#                     temp_new = signal.resample(temp_new, 3265)
#                 elif j == 'stimuli_13':
#                     ecg1_new = signal.resample(ecg1_new, 3714)
#                     ecg2_new = signal.resample(ecg2_new, 3714)
#                     gsr1_new = signal.resample(gsr1_new, 3714)
#                     gsr2_new = signal.resample(gsr2_new, 3714)
#                     erg1_new = signal.resample(erg1_new, 3714)
#                     erg2_new = signal.resample(erg2_new, 3714)
#                     resp_new = signal.resample(resp_new, 3714)
#                     plet_new = signal.resample(plet_new, 3714)
#                     temp_new = signal.resample(temp_new, 3714)
#
#                 rows = zip(ecg1_new, ecg2_new, gsr1_new, gsr2_new, erg1_new, erg2_new, resp_new, plet_new, temp_new)
#                 wr = csv.writer(out_file)
#                 if headers:
#                     wr.writerow(headers)    # returns the headers or `None` if the input is empty
#                 for row in rows:
#                     wr.writerow(row)

# resample Eye features
path = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Participant_Receiver/Eye/'
folders = os.listdir(path)
for i in folders:
    if i != '.DS_Store':
        print(i)
        os.chdir(path + i)
        for j in stimuli:
            with open(j + '.csv', 'r') as in_file, open(j + '_resampled.csv', 'w') as out_file:
                print(j + '.csv')
                file_content = csv.reader(in_file, delimiter=',')
                headers = next(file_content, None) # skip headers
                first_row = next(file_content, None) # remove first row because the row count is 1 more than the manual
                file_list = list(file_content)
                GED, FPX, FPY, GPLX, GPLY, GPRX, GPRY, GPX, GPY, GPX_m, GPY_m, GPLX_m, GPLY_m, GPRX_m, GPRY_m, EPLX,\
                EPLY, EPLZ, EPRX, EPRY, EPRZ, PL, PR, VL, VR  = ([] for i in range(25))

                for row in file_list:
                    GED.append(float(row[4]) if len(headers) == 33 else float(row[34]))
                    FPX.append(float(row[5]) if len(headers) == 33 else float(row[44]))
                    FPY.append(float(row[6]) if len(headers) == 33 else float(row[45]))
                    GPLX.append(float(row[11]) if len(headers) == 33 else float(row[29]))
                    GPLY.append(float(row[12]) if len(headers) == 33 else float(row[30]))
                    GPRX.append(float(row[13]) if len(headers) == 33 else float(row[31]))
                    GPRY.append(float(row[14]) if len(headers) == 33 else float(row[32]))
                    GPX.append(float(row[15]) if len(headers) == 33 else float(row[3]))
                    GPY.append(float(row[16]) if len(headers) == 33 else float(row[4]))
                    GPX_m.append(float(row[17]) if len(headers) == 33 else float(row[38]))
                    GPY_m.append(float(row[18]) if len(headers) == 33 else float(row[39]))
                    GPLX_m.append(float(row[19]) if len(headers) == 33 else float(row[25]))
                    GPLY_m.append(float(row[20]) if len(headers) == 33 else float(row[26]))
                    GPRX_m.append(float(row[21]) if len(headers) == 33 else float(row[27]))
                    GPRY_m.append(float(row[22]) if len(headers) == 33 else float(row[28]))
                    EPLX.append(float(row[23]) if len(headers) == 33 else float(row[19]))
                    EPLY.append(float(row[24]) if len(headers) == 33 else float(row[20]))
                    EPLZ.append(float(row[25]) if len(headers) == 33 else float(row[21]))
                    EPRX.append(float(row[26]) if len(headers) == 33 else float(row[22]))
                    EPRY.append(float(row[27]) if len(headers) == 33 else float(row[23]))
                    EPRZ.append(float(row[28]) if len(headers) == 33 else float(row[24]))
                    PL.append(float(row[29]) if len(headers) == 33 else float(row[15]))
                    PR.append(float(row[30]) if len(headers) == 33 else float(row[16]))
                    VL.append(float(row[31]) if len(headers) == 33 else float(row[17]))
                    VR.append(float(row[32]) if len(headers) == 33 else float(row[18]))

                GED_new = signal.decimate(GED, (2 if len(file_list) < 10000 else 10))
                FPX_new = signal.decimate(FPX, (2 if len(file_list) < 10000 else 10))
                FPY_new = signal.decimate(FPY, (2 if len(file_list) < 10000 else 10))
                GPLX_new = signal.decimate(GPLX, (2 if len(file_list) < 10000 else 10))
                GPLY_new = signal.decimate(GPLY, (2 if len(file_list) < 10000 else 10))
                GPRX_new = signal.decimate(GPRX, (2 if len(file_list) < 10000 else 10))
                GPRY_new = signal.decimate(GPRY, (2 if len(file_list) < 10000 else 10))
                GPX_new = signal.decimate(GPX, (2 if len(file_list) < 10000 else 10))
                GPY_new = signal.decimate(GPY, (2 if len(file_list) < 10000 else 10))
                GPX_m_new = signal.decimate(GPX_m, (2 if len(file_list) < 10000 else 10))
                GPY_m_new = signal.decimate(GPY_m, (2 if len(file_list) < 10000 else 10))
                GPLX_m_new = signal.decimate(GPLX_m, (2 if len(file_list) < 10000 else 10))
                GPLY_m_new = signal.decimate(GPLY_m, (2 if len(file_list) < 10000 else 10))
                GPRX_m_new = signal.decimate(GPRX_m, (2 if len(file_list) < 10000 else 10))
                GPRY_m_new = signal.decimate(GPRY_m, (2 if len(file_list) < 10000 else 10))
                EPLX_new = signal.decimate(EPLX, (2 if len(file_list) < 10000 else 10))
                EPLY_new = signal.decimate(EPLY, (2 if len(file_list) < 10000 else 10))
                EPLZ_new = signal.decimate(EPLZ, (2 if len(file_list) < 10000 else 10))
                EPRX_new = signal.decimate(EPRX, (2 if len(file_list) < 10000 else 10))
                EPRY_new = signal.decimate(EPRY, (2 if len(file_list) < 10000 else 10))
                EPRZ_new = signal.decimate(EPRZ, (2 if len(file_list) < 10000 else 10))
                PL_new = signal.decimate(PL, (2 if len(file_list) < 10000 else 10))
                PR_new = signal.decimate(PR, (2 if len(file_list) < 10000 else 10))
                VL_new = signal.decimate(VL, (2 if len(file_list) < 10000 else 10))
                VR_new = signal.decimate(VR, (2 if len(file_list) < 10000 else 10))

                rows = zip(GED_new, FPX_new, FPY_new, GPLX_new, GPLY_new, GPRX_new, GPRY_new, GPX_new, GPY_new, GPX_m_new, GPY_m_new, GPLX_m_new,
                           GPLY_m_new, GPRX_m_new, GPRY_m_new, EPLX_new, EPLY_new, EPLZ_new, EPRX_new, EPRY_new, EPRZ_new, PL_new, PR_new, VL_new, VR_new)
                wr = csv.writer(out_file)
                if headers:
                    wr.writerow(['GED', 'FPX', 'FPY', 'GPLX', 'GPLY', 'GPRX', 'GPRY', 'GPX', 'GPY', 'GPX_m', 'GPY_m', 'GPLX_m',
                                 'GPLY_m', 'GPRX_m', 'GPRY_m', 'EPLX', 'EPLY', 'EPLZ', 'EPRX', 'EPRY', 'EPRZ', 'PL', 'PR', 'VL', 'VR'])    # returns the headers or `None` if the input is empty
                for row in rows:
                    wr.writerow(row)
