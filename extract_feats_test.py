import os
import csv

par_eye = []
par_face = []
par_physio = []

par_headers = []

path_par_eye = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Test_dataset/Participant_Receiver/Eye/'
path_par_face = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Test_dataset/Participant_Receiver/Face/'
path_par_physio = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/Test_dataset/Participant_Receiver/Physio/'

files = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4', 'stimuli_5', 'stimuli_6', 'stimuli_7', 'stimuli_8',
         'stimuli_9', 'stimuli_10', 'stimuli_11', 'stimuli_12', 'stimuli_13']

# extract features

for folder in os.listdir(path_par_eye):
    if folder != '.DS_Store':
        print('Eye/' + folder)
        os.chdir(path_par_eye + folder)
        for file in files:
            print(file)
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
            print(file)
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
            print(file)
            with open(file + '_resampled.csv', 'r') as in_file:
                file_content = csv.reader(in_file, delimiter=',')
                headers_3 = next(file_content, None)
                for row in file_content:
                    par_physio.append(row)
par_headers += headers_3

print(len(par_physio))
print(par_headers)


# export csv files for participant

for i in range(10):
    rows2 = zip(par_eye[i*44923: (i+1)*44923], par_face[i*44923: (i+1)*44923], par_physio[i*44923: (i+1)*44923])
    with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/Test_dataset/' + str(i) + '.csv', 'w') as out_file2:
        wr2 = csv.writer(out_file2)
        wr2.writerow(par_headers)
        for row in rows2:
            flat_list = []
            for sublist in row:
                for item in sublist:
                    flat_list.append(item)
            wr2.writerow(flat_list)

print('feature extraction completed!')