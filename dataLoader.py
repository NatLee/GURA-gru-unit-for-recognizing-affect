import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from utils import cleanText, getPaddingSequence
from tqdm import tqdm

def loadAnimePickle(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load Anime data.')
    if not fromPickle:
        datasetName = 'dataset/animeReviewsSkipThoughtSummarization.pkl'
        with open(datasetName, 'rb') as p:
            X, Y = pickle.load(p)
        try:    
            X = getPaddingSequence([cleanText(text) for text in tqdm(X, ascii = True)], maxSeqLen, tokenizer)
            
            Ynew = np.asarray([ 1 if int(score)>=6 else 0 for score in Y], dtype='int8')
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

        with open('dataset/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'wb') as p:
            pickle.dump((X, Ynew), p)
    else:
        with open('dataset/animeReviewsSkipThoughtSummarizationPreprocessed.pkl', 'rb') as p:
            X, Ynew = pickle.load(p)
    return X, Ynew


def loadAmazon(maxSeqLen:int, tokenizer:Tokenizer):
    print('[INFO] Load Amazon data.')
    datasetName = 'dataset/AmazonFoodReviews.csv'
    df = pd.read_csv(datasetName, encoding='utf-8')
    Y = to_categorical(np.asarray(df['Score'], dtype='float16')-1, 5)
    X = getPaddingSequence([cleanText(text) for text in tqdm(df['Text'], ascii = True)], maxSeqLen, tokenizer)
    return X, Y

def loadIMDB(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load IMDB data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences = list()
        Y_train = list()
        with open('dataset/imdbReviews.csv', 'r', encoding='iso-8859-1') as f:
            f.readline() # filter the title
            for line in tqdm(f.readlines(), ascii = True):
                indices = [i for i, x in enumerate(line) if x == ',']
                dataType = line[indices[0]+1:indices[1]]
                label = line[indices[-2]+1:indices[-1]]
                if label == 'neg':
                    score = 0
                elif label == 'pos':
                    score = 1
                else:
                    continue
                text = cleanText(line[indices[1]+2:indices[-2]-1])
                if dataType=='test':
                    X_vali_Sentences.append(text)
                    Y_vali.append(score)
                elif dataType=='train':
                    X_train_Sentences.append(text)
                    Y_train.append(score)

        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/imdbReviewsPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/imdbReviewsPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)


    return X_vali, Y_vali, X_train, Y_train


def loadMPQA(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load MPQA data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences = list()
        Y_train = list()
        with open('dataset/MPQA.pkl', 'rb') as p:
            mpqa = pickle.load(p)
            sentences, labels, splits = list(mpqa.sentence), list(mpqa.label), list(mpqa.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    X_train_Sentences.append(cleanText(sentence))
                    Y_train.append(int(labels[i]))
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/MPQAPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/MPQAPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train

def loadMR(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load MR data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences = list()
        Y_train = list()
        with open('dataset/MR.pkl', 'rb') as p:
            mr = pickle.load(p)
            sentences, labels, splits = list(mr.sentence), list(mr.label), list(mr.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    X_train_Sentences.append(cleanText(sentence))
                    Y_train.append(int(labels[i]))
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/MRPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/MRPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train


def loadSST2(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load SST2 data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences = list()
        Y_train = list()
        with open('dataset/SST2.pkl', 'rb') as p:
            sst2 = pickle.load(p)
            sentences, labels, splits = list(sst2.sentence), list(sst2.label), list(sst2.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    X_train_Sentences.append(cleanText(sentence))
                    Y_train.append(int(labels[i]))
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/SST2Preprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/SST2Preprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_vali, Y_vali, X_train, Y_train


def loadSUBJ(maxSeqLen:int, tokenizer:Tokenizer, fromPickle:bool):
    print('[INFO] Load SUBJ data.')
    if not fromPickle:
        X_vali_Sentences = list()
        Y_vali = list()
        X_train_Sentences = list()
        Y_train = list()
        with open('dataset/SUBJ.pkl', 'rb') as p:
            subj = pickle.load(p)
            sentences, labels, splits = list(subj.sentence), list(subj.label), list(subj.split)
            for i, sentence in enumerate(tqdm(sentences, ascii = True)):
                if splits[i] == 'train':
                    X_train_Sentences.append(cleanText(sentence))
                    Y_train.append(int(labels[i]))
                else:
                    X_vali_Sentences.append(cleanText(sentence))
                    Y_vali.append(int(labels[i]))

        X_vali = getPaddingSequence(X_vali_Sentences, maxSeqLen, tokenizer)
        X_train = getPaddingSequence(X_train_Sentences, maxSeqLen, tokenizer)

        with open('dataset/SUBJPreprocessed.pkl', 'wb') as p:
            pickle.dump((X_vali, Y_vali, X_train, Y_train), p)
    else:
        with open('dataset/SUBJPreprocessed.pkl', 'rb') as p:
            X_vali, Y_vali, X_train, Y_train = pickle.load(p)

    return X_train, Y_train