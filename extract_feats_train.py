import os
import csv

par_eye = []
par_face = []
par_physio = []
sti_audio = []
sti_face_eye = []
competence = []
warmth = []

par_headers = []
sti_headers = []
lab_headers = []

path_par_eye = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Participant_Receiver/Eye/'
path_par_face = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Participant_Receiver/Face/'
path_par_physio = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Participant_Receiver/Physio/'
path_sti_audio = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Stimuli_Emitter/Audio/'
path_sti_face_eye = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Stimuli_Emitter/Face&Eye/'
path_competence = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Label/competence/'
path_warmth = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Training_dataset/Label/warmth/'

files = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4', 'stimuli_5', 'stimuli_6', 'stimuli_7', 'stimuli_8',
         'stimuli_9', 'stimuli_10', 'stimuli_11', 'stimuli_12', 'stimuli_13']
emitters = ['4', '6', '7', '9', '10', '11', '13', '14', '15', '16', '18', '20', '21']

# extract features

for folder in os.listdir(path_par_eye):
    if folder != '.DS_Store':
        print('Eye/' + folder)
        os.chdir(path_par_eye + folder)
        for file in files:
            with open(file + '_resampled.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_1 = next(file_content, None)
                for row in file_content:
                    par_eye.append(row)
par_headers += headers_1

print(len(par_eye))
print(par_headers)


for folder in os.listdir(path_par_face):
    if folder != '.DS_Store':
        print('Face/' + folder)
        os.chdir(path_par_face + folder)
        for file in files:
            # print(file)
            with open(file + '.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_2 = next(file_content, None)
                for row in file_content:
                    par_face.append(row[-35:])
par_headers += headers_2[-35:]

print(len(par_face))
print(par_headers)


for folder in os.listdir(path_par_physio):
    if folder != '.DS_Store':
        print('Physio/' + folder)
        os.chdir(path_par_physio + folder)
        for file in files:
            # print(file)
            with open(file + '_resampled.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_3 = next(file_content, None)
                for row in file_content:
                    par_physio.append(row)
par_headers += headers_3

print(len(par_physio))
print(par_headers)


os.chdir(path_sti_audio)
for file in emitters:
    print('Audio/' + file)
    with open(file + '_resample.csv', 'r') as in_file:
        file_content = csv.reader(in_file, delimiter=',')
        headers_4 = next(file_content, None)
        for row in file_content:
            sti_audio.append(row)
sti_headers += headers_4

print(len(sti_audio))
print(sti_headers)


os.chdir(path_sti_face_eye)
for file in emitters:
    print('Face&Eye/' + file + '.csv')
    with open(file + '.au_class.csv', 'r') as in_file1, open(file + '.au_reg.csv', 'r') as in_file2, \
            open(file + '.landmarks_2d.csv', 'r') as in_file3, open(file + '.landmarks_3d.csv', 'r') as in_file4, \
            open(file + '.params.csv', 'r') as in_file5, open(file + '.pose.csv', 'r') as in_file6:
        au_c = csv.reader(in_file1, delimiter=',')
        au_r = csv.reader(in_file2, delimiter=',')
        la_2 = csv.reader(in_file3, delimiter=',')
        la_3 = csv.reader(in_file4, delimiter=',')
        para = csv.reader(in_file5, delimiter=',')
        pose = csv.reader(in_file6, delimiter=',')

        headers_5 = next(au_c, None)
        headers_6 = next(au_r, None)
        headers_7 = next(la_2, None)
        headers_8 = next(la_3, None)
        headers_9 = next(para, None)
        headers_10 = next(pose, None)

        file_list1 = list(au_c)
        file_list2 = list(au_r)
        file_list3 = list(la_2)
        file_list4 = list(la_3)
        file_list5 = list(para)
        file_list6 = list(pose)

        i = 0
        # print(len(sti_face_eye))
        while i < len(file_list1):
            sti_face_eye.append(file_list1[i][3:] + file_list2[i][3:] + file_list3[i][3:] + file_list4[i][3:]
                                + file_list5[i][9:] + file_list6[i][6:])
            i += 1

sti_headers += headers_5[3:]
sti_headers += headers_6[3:]
sti_headers += headers_7[3:]
sti_headers += headers_8[3:]
sti_headers += headers_9[9:]
sti_headers += headers_10[6:]

print(len(sti_face_eye))
print(sti_headers)


# extract labels

for folder in os.listdir(path_competence):
    if folder != '.DS_Store':
        # print('competence/' + folder)
        os.chdir(path_competence + folder)
        for file in files:
            # print(file)
            with open(file + '.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_11 = next(file_content, None)
                for row in file_content:
                    competence.append(row[1])
lab_headers += [headers_11[1]]

print(len(competence))
print(lab_headers)


for folder in os.listdir(path_warmth):
    if folder != '.DS_Store':
        # print('warmth/' + folder)
        os.chdir(path_warmth + folder)
        for file in files:
            # print(file)
            with open(file + '.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_12 = next(file_content, None)
                for row in file_content:
                    warmth.append(row[1])
lab_headers += [headers_12[1]]

print(len(warmth))
print(lab_headers)

# export csv files for stimuli

rows1 = zip(sti_audio, sti_face_eye)
with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_stimuli/' + 'sti.csv', 'w') as out_file1:
    wr1 = csv.writer(out_file1)
    wr1.writerow(sti_headers)
    for row in rows1:
        flat_list = []
        for sublist in row:
            for item in sublist:
                flat_list.append(item)
        wr1.writerow(flat_list)

# export csv files for participant

for i in range(40):
    rows2 = zip(par_eye[i*44923: (i+1)*44923], par_face[i*44923: (i+1)*44923], par_physio[i*44923: (i+1)*44923])
    rows3 = zip(competence[i*44923: (i+1)*44923], warmth[i*44923: (i+1)*44923])
    with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_participant/' + str(i) + '.csv', 'w') as out_file2:
        wr2 = csv.writer(out_file2)
        wr2.writerow(par_headers)
        for row in rows2:
            flat_list = []
            for sublist in row:
                for item in sublist:
                    flat_list.append(item)
            wr2.writerow(flat_list)

    with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/labels/' + str(i) + '.csv', 'w') as out_file3:
        wr3 = csv.writer(out_file3)
        wr3.writerow(lab_headers)
        for row in rows3:
            wr3.writerow(row)

print('feature extraction completed!')