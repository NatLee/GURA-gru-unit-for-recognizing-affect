import json
from pathlib import Path
from subprocess import check_output

TARGET_FLAG = True
accHistoryFilePathString = 'accHistory.json'

imdbTargetAcc = 0.8905
sst2TargetAcc = 0.8300
mpqaTargetAcc = 0.8814
mrTargetAcc = 0.8700
subjTargetAcc = 0.9150
animeTargetAcc = 0.8500

def loadHistory(path):
    
    def getDataset(history, name):
        valAcc = history.get(name)
        if valAcc is None:
            return 0
        else:
            return valAcc

    path = Path(path)
    if path.exists():
        with open(path.absolute().as_posix(), 'r') as f:
            accHistory = json.load(f)
    
            imdbValAcc = getDataset(accHistory, 'imdb')
            sst2ValAcc = getDataset(accHistory, 'sst2')
            mpqaValAcc = getDataset(accHistory, 'mpqa')
            mrValAcc = getDataset(accHistory, 'mr')
            subjValAcc = getDataset(accHistory, 'subj')
            animeValAcc = getDataset(accHistory, 'anime')
    else:
        imdbValAcc = 0
        sst2ValAcc = 0
        mpqaValAcc = 0
        mrValAcc = 0
        subjValAcc = 0
        animeValAcc = 0

    return imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc


imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc = loadHistory(accHistoryFilePathString)

#datasets = ['--imdb ', '--sst2 ', '--mpqa ', '--mr ', '--subj ', '--anime ']
datasets = ['--anime ']

while TARGET_FLAG:

    print('[INFO] IMDB TOP VAL ACC: {:.2f}'.format(imdbValAcc))
    print('[INFO] SST2 TOP VAL ACC: {:.2f}'.format(sst2ValAcc))
    print('[INFO] MPQA TOP VAL ACC: {:.2f}'.format(mpqaValAcc))
    print('[INFO] MR TOP VAL ACC: {:.2f}'.format(mrValAcc))
    print('[INFO] SUBJ TOP VAL ACC: {:.2f}'.format(subjValAcc))
    print('[INFO] ANIME TOP VAL ACC: {:.2f}'.format(animeValAcc))

    runCommand = ''
    for dataset in datasets:
        if imdbValAcc > imdbTargetAcc and dataset.find('imdb') >= 0:
            continue
        if sst2ValAcc > sst2TargetAcc and dataset.find('sst2') >= 0:
            continue
        if mpqaValAcc > mpqaTargetAcc and dataset.find('mpqa') >= 0:
            continue
        if mrValAcc > mrTargetAcc and dataset.find('mr') >= 0:
            continue
        if subjValAcc > subjTargetAcc and dataset.find('subj') >= 0:
            continue
        if animeValAcc > animeTargetAcc and dataset.find('anime') >= 0:
            continue

        runCommand = runCommand + dataset

    if runCommand == '':
        TARGET_FLAG = False
    else:
        check_output('python main.py ' + runCommand, shell=True).decode()

        imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc = loadHistory(accHistoryFilePathString)

    