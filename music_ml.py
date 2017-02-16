import librosa
import numpy as np
import random
import os
import sklearn

filetypes = ['.wav', '.mp3', '.flac', '.ogg']

sr = 22050

def load_files(*file_lists, train_pct=33, test_pct=33):
    """returns 2 dicts: train_files, and test_files"""

    p_dict = {}
    
    for i in range(len(file_lists)):
        pl = _parse_list(file_lists[i])
        for _ in pl:
            p_dict[_] = {}
            p_dict[_]['songname'] = os.path.splitext((os.path.split(_)[1]))[0]
            p_dict[_]['data'] = _load_data(_)
            p_dict[_]['categorization'] = i
            #print(p_dict[_])

    train_files, test_files = _split_files(train_pct, test_pct, p_dict)

    print(str(len(train_files)) + " training songs, " + str(len(test_files)) + " testing songs")
    return train_files, test_files        

def _split_files(train_pct, test_pct, p_dict):
    """Splits available files into test and training sets"""
    #FIX: Make sure all classifications are represented, as best as possible
    train_files, test_files = {}, {}
    if train_pct + test_pct > 100:
        print("Error: Total training and testing percentages should total less than 100")
    else:
        while len(train_files.keys()) < int(len(p_dict.keys())*train_pct*.01):
            r = random.choice(list(p_dict.keys()))
            train_files[r] = p_dict[r]


        while len(test_files.keys()) < int(len(p_dict.keys())*test_pct*.01):
            r = random.choice(list(p_dict.keys()))
            test_files[r] = p_dict[r]
        
    return train_files, test_files

def _parse_list(flist):
    """Does basic cleaning on a list of files"""
    parsed_list = []

    if type(flist) != list:
        print("Error: Expecting a list or lists of files")

    for f in flist:
        if not os.path.exists(f):
            print("Error: File " + f + " does not exist")
            continue
        if os.path.isdir(f):
            dir_files = [os.path.join(f, _) for _ in os.listdir(f) if os.path.isfile(os.path.join(f, _))]
            flist.extend(dir_files)
        elif f not in parsed_list:
            parsed_list.append(f)

    parsed_ft = [_ for _ in parsed_list if os.path.splitext(_)[-1].lower() in filetypes]
    
    return parsed_ft
    
def _load_data(fname):
    print("Loading file: " + fname)
    x,_ = librosa.load(fname)
    #x = [1,2,3,4,5]
    return x

def calculate_features(save_file, file_list, *feature_funcs):
    """Calculates features for each file in file_list, using all passed feature_funcs. Returns file_list dict with appended feature:value k:v pairs"""

    for k,v in file_list.items():
        print("Calculating features for " + file_list[k]['songname'])
        for ff in feature_funcs:
            ff_name, ff_value = ff(file_list[k]['data'])
            print(ff_name + " calculated")
            file_list[k][ff_name] = ff_value

    return file_list

def ff_bpm(song_data):
    """Calculates BPM for a song"""
    onset_env = librosa.onset.onset_strength(song_data, sr=sr)
    return ('BPM', librosa.beat.estimate_tempo(onset_env, sr=sr))

def ff_mfcc(song_data, n=40):
    """Calculates n MFCCs for a song"""
    mfcc_mat = librosa.feature.mfcc(y=song_data, sr=sr)

    #Collapse across each time-series
    flattened_mat = [sum(row)/float(len(row)) for row in mfcc_mat]

    return ('MFCC' + str(n), flattened_mat)

##def ml_knn(train_files, test_files, k=0):
##    """Calculates k nearest neighbors"""
##
##    if k == 0:
##        #If no value is passed, k should be equal to the number of different classifications
##        k = len(list(set([train_files[k]['categorization'] for k,v in train_files.items()])))
##
##    clf = sklearn.neighbors.KNeighborsClassifier(k, weights=weights)
##    clf.fit(X, y)

def ml_svm(train_files, test_files):
    """SVM using MFCCs"""
    pass
