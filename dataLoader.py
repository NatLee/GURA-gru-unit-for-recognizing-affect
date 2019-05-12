import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from utils import cleanText, getPaddingSequence
from tqdm import tqdm
from random import randint


def genData(x:list, polarity:int, number:int=None):

    print('[INFO] Generate data...')

    avgSentenLength = sum([len(sen.split()) for sen in x])/len(x)
    swd = ' '.join(x).split()
    gx = list()
    gy = list()

    if number is None:
        number = len(x)

    for i in tqdm(range(number), ascii=True):
        sentence = list()
        randomLength = avgSentenLength + randint(-5, 5)
        randomLength = 128 if randomLength >128 else randomLength
        randomLength = avgSentenLength if randomLength <0 else randomLength
        while len(sentence) < randomLength:
            rn = randint(0, len(swd)-1)
            s = randint(-8, 0)
            e = randint(0, 8)
            for i in range(s, e):
                b = rn + i
                if len(swd)-1< (b):
                    continue
                sentence.append(swd[b])
        gx.append(' '.join(sentence))
        gy.append(polarity)
    return gx, gy


def loadAnimePickle(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    print('[INFO] Load Anime data.')
    if not fromPickle:
        datasetName = 'dataset/animeReviewsSkipThoughtSummarization.pkl'
        with open(datasetName, 'rb') as p:
            X, Y = pickle.load(p)

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


            print('[INFO] Original positive: {}'.format(len(X_pos)))
            print('[INFO] Original negative: {}'.format(len(X_neg)))


            gx_pos, gy_pos = genData(X_pos, 1, 3899)
            gx_neg, gy_neg = genData(X_neg, 0, 77755)


            X = X_pos + X_neg + gx_pos + gx_neg
            Y = Y_pos + Y_neg + gy_pos + gy_neg

            X = getPaddingSequence(X, maxSeqLen, tokenizer)            
            Y = np.asarray(Y)


        with open('dataset/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'wb') as p:
            pickle.dump((X, Y), p)
    else:
        with open('dataset/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'rb') as p:
            X, Y = pickle.load(p)
    return X, Y


def loadAmazon(maxSeqLen:int, tokenizer:Tokenizer):
    print('[INFO] Load Amazon data.')
    datasetName = 'dataset/AmazonFoodReviews.csv'
    df = pd.read_csv(datasetName, encoding='utf-8')
    Y = to_categorical(np.asarray(df['Score'], dtype='float16')-1, 5)
    X = getPaddingSequence([cleanText(text) for text in tqdm(df['Text'], ascii = True)], maxSeqLen, tokenizer)
    return X, Y


def loadOfficialIMDB(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):

    print('[INFO] Load official IMDB data.')

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
        imdb = Path('dataset/aclImdb/')

        pos = (imdb / folder / 'pos').glob('./*.txt')
        neg = (imdb / folder / 'neg').glob('./*.txt')

        posFiles = [x for x in pos if x.is_file()]
        negFiles = [x for x in neg if x.is_file()]

        print('[INFO] Original positive: {}'.format(len(posFiles)))
        print('[INFO] Original negative: {}'.format(len(negFiles)))

        #print('[INFO] -----Loading positive data')
        x_pos, y_pos = loadFiles(posFiles, 1, dataClass)

        #print('[INFO] -----Loading negative data')
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


        with open('dataset/imdbOfficailReviewsPreprocessed.pkl', 'wb') as p:
            pickle.dump((x_test, y_test, x_train, y_train), p)
    else:
        with open('dataset/imdbOfficailReviewsPreprocessed.pkl', 'rb') as p:
            x_test, y_test, x_train, y_train = pickle.load(p)

    return x_test, y_test, x_train, y_train


def loadMPQA(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    print('[INFO] Load MPQA data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('dataset/MPQA.pkl', 'rb') as p:
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

        print('[INFO] Original positive: {}'.format(len(X_train_Sentences_pos)))
        print('[INFO] Original negative: {}'.format(len(X_train_Sentences_neg)))

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 6688)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 2706)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/MPQAPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/MPQAPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train

def loadMR(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    print('[INFO] Load MR data.')
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

        print('[INFO] Original positive: {}'.format(len(X_train_Sentences_pos)))
        print('[INFO] Original negative: {}'.format(len(X_train_Sentences_neg)))

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 4669)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 4669)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/MRPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/MRPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train


def loadSST2(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    print('[INFO] Load SST2 data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('dataset/SST2.pkl', 'rb') as p:
            sst2 = pickle.load(p)
            sentences, labels, splits = list(sst2.sentence), list(sst2.label), list(sst2.split)
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


        print('[INFO] Original positive: {}'.format(len(X_train_Sentences_pos)))
        print('[INFO] Original negative: {}'.format(len(X_train_Sentences_neg)))

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1, 7741)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0, 15298)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/SST2Preprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/SST2Preprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train


def loadSUBJ(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool, number:int=None):
    print('[INFO] Load SUBJ data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences_pos = list()
        X_train_Sentences_neg = list()
        Y_train_pos = list()
        Y_train_neg = list()
        with open('dataset/SUBJ.pkl', 'rb') as p:
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

        print('[INFO] Original positive: {}'.format(len(X_train_Sentences_pos)))
        print('[INFO] Original negative: {}'.format(len(X_train_Sentences_neg)))

        gx_pos, gy_pos = genData(X_train_Sentences_pos, 1)
        gx_neg, gy_neg = genData(X_train_Sentences_neg, 0)

        X_train_Sentences = X_train_Sentences_pos + X_train_Sentences_neg + gx_pos + gx_neg
        Y_train = Y_train_pos + Y_train_neg + gy_pos + gy_neg



        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/SUBJPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/SUBJPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train