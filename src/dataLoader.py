import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from utils import cleanText, getPaddingSequence
from tqdm import tqdm
from random import randint, shuffle


def randomPick(xList, yList, pickNumber):
    randomData = list(zip(xList, yList))
    shuffle(randomData)
    validX = list()
    validY = list()
    trainX = list()
    trainY = list()
    for x, y in randomData[:pickNumber]:
        validX.append(x)
        validY.append(y)
    for x, y in randomData[pickNumber:]:
        trainX.append(x)
        trainY.append(y)

    return trainX, trainY, validX, validY

def genData(x:list, polarity:int, number:int=None):

    logging.info('Generate data...')

    avgSentenLength = sum([len(sen.split()) for sen in x])/len(x)
    logging.info('Average sentence length: {}'.format(avgSentenLength))
    swd = ' '.join(x).split()
    gx = list()
    gy = list()

    if number is None:
        number = len(x)

    for i in tqdm(range(number), ascii=True):
        sentence = list()
        # one review length
        randomLength = avgSentenLength + randint(-5, 5)
        randomLength = 128 if randomLength >128 else randomLength
        randomLength = avgSentenLength if randomLength <0 else randomLength
        while len(sentence) < randomLength:
            rn = randint(0, len(swd)-1)
            # one sentence length
            s = randint(-8, 0)
            e = randint(0, 8)
            for i in range(s, e):
                b = rn + i
                if len(swd)-1 < b:
                    continue
                sentence.append(swd[b])
        gx.append(' '.join(sentence))
        gy.append(polarity)
    return gx, gy

def loadAnimePickle(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    logging.info('Load Anime data.')
    if not fromPickle:
        datasetName = '/data/animeReviewsSkipThoughtSummarization.pkl'
        with open(datasetName, 'rb') as p:
            X, Y = pickle.load(p)

            # Take before 2017
            X = X[:87415]
            Y = Y[:87415]

            X_pos = list()
            X_neg = list()
            Y_pos = list()
            Y_neg = list()

            for i, score in enumerate(tqdm(Y, ascii=True)):
                text = cleanText(X[i])

                if int(score)>=6:
                    X_pos.append(text)
                    Y_pos.append(1)
                else:
                    X_neg.append(text)
                    Y_neg.append(0)


            logging.info('Original Train positive: {}'.format(len(X_pos)))
            logging.info('Original Train negative: {}'.format(len(X_neg)))


            X_pos, Y_pos, X_vali_pos, Y_vali_pos = randomPick(X_pos, Y_pos, 14552)
            X_neg, Y_neg, X_vali_neg, Y_vali_neg = randomPick(X_neg, Y_neg, 2931)

            logging.info('Valid positive: {}'.format(len(X_vali_pos)))
            logging.info('Valid negative: {}'.format(len(X_vali_neg)))
            X_vali = X_vali_pos + X_vali_neg
            Y_vali = Y_vali_pos + Y_vali_neg


            gx_pos, gy_pos = genData(X_pos, 1, 1790)
            gx_neg, gy_neg = genData(X_neg, 0, 48278)


            X = X_pos + X_neg + gx_pos + gx_neg
            Y = Y_pos + Y_neg + gy_pos + gy_neg

            X = getPaddingSequence(X, maxSeqLen, tokenizer)         
            Y = np.asarray(Y)

            X_vali = getPaddingSequence(X_vali, maxSeqLen, tokenizer)
            Y_vali = np.asarray(Y_vali)


        with open('/data/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X, Y), p)
    else:
        with open('/data/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X, Y = pickle.load(p)

    return X_vali, Y_vali, X, Y

def loadOfficialIMDB(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):

    logging.info('Load official IMDB data.')

    def loadFiles(files:list, polarity:int, dataClass:str):
        x = list()
        y = list()

        for txtfile in tqdm(files, ascii=True):
            with open(txtfile, 'r', encoding='utf-8') as f:
                x.append(cleanText(f.read()))
                y.append(polarity)

        
        
        if dataClass == 'train':
            gx, gy = genData(x, polarity, number)
            x = x + gx
            y = y + gy
        

        return x, y

    def loadFromFolder(folder:str, dataClass:str=None):
        imdb = Path('/data/aclImdb/')

        pos = (imdb / folder / 'pos').glob('./*.txt')
        neg = (imdb / folder / 'neg').glob('./*.txt')

        posFiles = [x for x in pos if x.is_file()]
        negFiles = [x for x in neg if x.is_file()]

        logging.info('Original positive: {}'.format(len(posFiles)))
        logging.info('Original negative: {}'.format(len(negFiles)))

        #logging.info('-----Loading positive data')
        x_pos, y_pos = loadFiles(posFiles, 1, dataClass)

        #logging.info('-----Loading negative data')
        x_neg, y_neg = loadFiles(negFiles, 0, dataClass)

        x = x_pos + x_neg
        y = y_pos + y_neg
        
        return x, y

    if not fromPickle:
    
        print('------[INFO] Load testing set.')
        x_test, y_test = loadFromFolder('test')

        print('------[INFO] Load training set.')
        x_train, y_train = loadFromFolder('train', 'train')


        x_test = getPaddingSequence(x_test, maxSeqLen, tokenizer)
        x_train = getPaddingSequence(x_train, maxSeqLen, tokenizer)

        y_test = np.asarray(y_test)
        y_train = np.asarray(y_train)

        #y_one_hot_test = to_categorical(y_test, num_classes=2)
        #y_one_hot_train = to_categorical(y_train, num_classes=2)


        with open('/data/imdbOfficailReviewsPreprocessed.pkl', 'wb') as p:
            pickle.dump((x_test, y_test, x_train, y_train), p)
    else:
        with open('/data/imdbOfficailReviewsPreprocessed.pkl', 'rb') as p:
            x_test, y_test, x_train, y_train = pickle.load(p)

    return x_test, y_test, x_train, y_train

def loadMPQA(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    logging.info('Load MPQA data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('/data/MPQA.pkl', 'rb') as p:
            mpqa = pickle.load(p)
            sentences, labels, splits = list(mpqa.sentence), list(mpqa.label), list(mpqa.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    label = int(labels[i])
                    sentence = cleanText(sentence)
                    if label == 1:
                        X_train_Sentences_pos.append(sentence)
                        Y_train_pos.append(label)
                    else:
                        X_train_Sentences_neg.append(sentence)
                        Y_train_neg.append(label)
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        logging.info('Original Train positive: {}'.format(len(X_train_Sentences_pos)))
        logging.info('Original Train negative: {}'.format(len(X_train_Sentences_neg)))

        X_train_Sentences_pos, Y_train_pos, X_vali_Sentences_pos, Y_vali_pos = randomPick(X_train_Sentences_pos, Y_train_pos, 2700)
        X_train_Sentences_neg, Y_train_neg, X_vali_Sentences_neg, Y_vali_neg = randomPick(X_train_Sentences_neg, Y_train_neg, 6400)

        logging.info('Valid positive: {}'.format(len(X_vali_Sentences_pos)))
        logging.info('Valid negative: {}'.format(len(X_vali_Sentences_neg)))

        X_vali_Sentences = X_vali_Sentences + X_vali_Sentences_pos + X_vali_Sentences_neg
        Y_vali = Y_vali + Y_vali_pos + Y_vali_neg


        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 6688)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 2706)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('/data/MPQAPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('/data/MPQAPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train

def loadMR(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    logging.info('Load MR data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('dataset/MR.pkl', 'rb') as p:
            mr = pickle.load(p)
            sentences, labels, splits = list(mr.sentence), list(mr.label), list(mr.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    label = int(labels[i])
                    sentence = cleanText(sentence)
                    if label == 1:
                        X_train_Sentences_pos.append(sentence)
                        Y_train_pos.append(label)
                    else:
                        X_train_Sentences_neg.append(sentence)
                        Y_train_neg.append(label)
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))


        logging.info('Original Train positive: {}'.format(len(X_train_Sentences_pos)))
        logging.info('Original Train negative: {}'.format(len(X_train_Sentences_neg)))

        X_train_Sentences_pos, Y_train_pos, X_vali_Sentences_pos, Y_vali_pos = randomPick(X_train_Sentences_pos, Y_train_pos, 4500)
        X_train_Sentences_neg, Y_train_neg, X_vali_Sentences_neg, Y_vali_neg = randomPick(X_train_Sentences_neg, Y_train_neg, 4500)

        logging.info('Valid positive: {}'.format(len(X_vali_Sentences_pos)))
        logging.info('Valid negative: {}'.format(len(X_vali_Sentences_neg)))

        X_vali_Sentences = X_vali_Sentences + X_vali_Sentences_pos + X_vali_Sentences_neg
        Y_vali = Y_vali + Y_vali_pos + Y_vali_neg


        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 6669)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 6669)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('/data/MRPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('/data/MRPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train

def loadSST2(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    logging.info('Load SST2 data.')
    if not fromPickle:
        X_vali_Sentences_pos = list()
        X_vali_Sentences_neg = list()
        Y_vali_pos = list()
        Y_vali_neg = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('dataset/SST2.pkl', 'rb') as p:
            sst2 = pickle.load(p)
            sentences, labels, splits = list(sst2.sentence), list(sst2.label), list(sst2.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                label = int(labels[i])
                sentence = cleanText(sentence)
                if splits[i] == 'train':    
                    if label == 1:
                        X_train_Sentences_pos.append(sentence)
                        Y_train_pos.append(label)
                    else:
                        X_train_Sentences_neg.append(sentence)
                        Y_train_neg.append(label)
                else:
                    if label == 1:
                        X_vali_Sentences_pos.append(sentence)
                        Y_vali_pos.append(label)
                    else:
                        X_vali_Sentences_neg.append(sentence)
                        Y_vali_neg.append(label)


        logging.info('Original Train positive: {}'.format(len(X_train_Sentences_pos)))
        logging.info('Original Train negative: {}'.format(len(X_train_Sentences_neg)))


        logging.info('Valid positive: {}'.format(len(X_vali_Sentences_pos)))
        logging.info('Valid negative: {}'.format(len(X_vali_Sentences_neg)))
        X_vali_Sentences = X_vali_Sentences_pos + X_vali_Sentences_neg
        Y_vali = Y_vali_pos + Y_vali_neg

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 7741)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 15298)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('/data/SST2Preprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('/data/SST2Preprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train

def loadSUBJ(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):

    logging.info('Load SUBJ data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('/data/SUBJ.pkl', 'rb') as p:
            subj = pickle.load(p)
            sentences, labels, splits = list(subj.sentence), list(subj.label), list(subj.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    label = int(labels[i])
                    sentence = cleanText(sentence)
                    if label == 1:
                        X_train_Sentences_pos.append(sentence)
                        Y_train_pos.append(label)
                    else:
                        X_train_Sentences_neg.append(sentence)
                        Y_train_neg.append(label)
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        logging.info('Original Train positive: {}'.format(len(X_train_Sentences_pos)))
        logging.info('Original Train negative: {}'.format(len(X_train_Sentences_neg)))

        X_train_Sentences_pos, Y_train_pos, X_vali_Sentences_pos, Y_vali_pos = randomPick(X_train_Sentences_pos, Y_train_pos, 4500)
        X_train_Sentences_neg, Y_train_neg, X_vali_Sentences_neg, Y_vali_neg = randomPick(X_train_Sentences_neg, Y_train_neg, 4500)

        logging.info('Valid positive: {}'.format(len(X_vali_Sentences_pos)))
        logging.info('Valid negative: {}'.format(len(X_vali_Sentences_neg)))

        X_vali_Sentences = X_vali_Sentences + X_vali_Sentences_pos + X_vali_Sentences_neg
        Y_vali = Y_vali + Y_vali_pos + Y_vali_neg

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('/data/SUBJPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('/data/SUBJPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train

