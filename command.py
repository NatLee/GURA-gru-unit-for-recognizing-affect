import json
from pathlib import Path
from subprocess import check_output


FLAG = True
accHistoryFilePathString = 'accHistory.json'

def loadHistory(path):
    path = Path(path)
    if path.exists():
        with open(path.absolute().as_posix(), 'r') as f:
            accHistory = json.load(f)

            imdbValAcc = accHistory['imdb']
            sst2ValAcc = accHistory['sst2']
            mpqaValAcc = accHistory['mpqa']
            mrValAcc = accHistory['mr']
            subjValAcc = accHistory['subj']
            animeValAcc = accHistory['anime']
    else:
        imdbValAcc = 0
        sst2ValAcc = 0
        mpqaValAcc = 0
        mrValAcc = 0
        subjValAcc = 0
        animeValAcc = 0

    return imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc


imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc = loadHistory(accHistoryFilePathString)

datasets = ['--imdb ', '--sst2 ', '--mpqa ', '--mr ', '--subj ', '--anime ']

while FLAG:

    print('[INFO] IMDB TOP VAL ACC: {:.2f}'.format(imdbValAcc))
    print('[INFO] SST2 TOP VAL ACC: {:.2f}'.format(sst2ValAcc))
    print('[INFO] MPQA TOP VAL ACC: {:.2f}'.format(mpqaValAcc))
    print('[INFO] MR TOP VAL ACC: {:.2f}'.format(mrValAcc))
    print('[INFO] SUBJ TOP VAL ACC: {:.2f}'.format(subjValAcc))
    print('[INFO] ANIME TOP VAL ACC: {:.2f}'.format(animeValAcc))

    runCommand = ''
    for dataset in datasets:
        if imdbValAcc > 0.8905 and dataset.find('imdb') >= 0:
            continue
        if sst2ValAcc > 0.8300 and dataset.find('sst2') >= 0:
            continue
        if mpqaValAcc > 0.8800 and dataset.find('mpqa') >= 0:
            continue
        if mrValAcc > 0.8700 and dataset.find('mr') >= 0:
            continue
        if subjValAcc > 0.9150 and dataset.find('subj') >= 0:
            continue
        if animeValAcc > 0.8500 and dataset.find('anime') >= 0:
            continue

        runCommand = runCommand + dataset

    if runCommand == '':
        FLAG = False
    else:
        check_output('python main.py ' + runCommand, shell=True).decode()

        imdbValAcc, sst2ValAcc, mpqaValAcc, mrValAcc, subjValAcc, animeValAcc = loadHistory(accHistoryFilePathString)