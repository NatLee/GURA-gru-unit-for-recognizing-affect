import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import numpy as np


def saveTrainingImg(historyRecord:dict, savePath:str) -> bool:
    acc = historyRecord.history['acc']
    val_acc = historyRecord.history['val_acc']
    loss = historyRecord.history['loss']
    val_loss = historyRecord.history['val_loss']

    epochs = historyRecord.epoch

    colors = ['#2300A8', '#00A658']
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax1.plot(epochs, loss, marker='.', linestyle='dashed', alpha=0.6, color=colors[0], label='Training loss')
    ax1.plot(epochs, val_loss, color=colors[0], label='Validation loss')
    #ax1.set_title('Loss')
    #ax1.set_xlabel('Epoch(s)')
    #ax1.set_xticks(np.arange(0, epochs, 1))
    #ax1.set_ylim(0,1)
    #ax1.set_yticks(np.arange(0, 1, 0.1))
    ax1.set_ylabel('Loss')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax1.legend()

    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.plot(epochs, acc, marker='.', linestyle='dashed', alpha=0.6, color=colors[1], label='Training acc')
    ax2.plot(epochs, val_acc, color=colors[1], label='Validation acc')
    #ax2.set_title('Accuracy')
    #ax2.set_xticks(np.arange(0, epochs, 1))
    ax2.set_xlabel('Epoch(s)')
    ax2.set_ylim(0,1)
    ax2.set_yticks(np.arange(0, 1, 0.1))
    ax2.set_ylabel('Acc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax2.legend()

    plt.savefig(savePath)
    return True


def cleanText(text, remove_stopwords=True, perform_stemming=True):

    '''
    #regex for removing non-alphanumeric characters and spaces
    remove_special_char = re.compile('r[^a-z\d]', re.IGNORECASE)
    #regex to replace all numerics
    replace_numerics = re.compile(r'\d+', re.IGNORECASE)
    text = remove_special_char.sub('', text)
    text = replace_numerics.sub('', text)
    '''

    stop_words = set(stopwords.words('english')) 
    stemmer = SnowballStemmer('english')

    #convert text to lowercase.
    text = text.lower().split()

    #lemmatize the text
    #lemmatizer = WordNetLemmatizer()
    #text = [lemmatizer.lemmatize(token) for token in text]
    #text = [lemmatizer.lemmatize(token, 'v') for token in text]
        
    processedText = list()
    for word in text:        
        if remove_stopwords:
            if word in stop_words:
                continue
        if perform_stemming:
            word = stemmer.stem(word)
        processedText.append(word)

    text = ' '.join(processedText)

    return text

def getPaddingSequence(wordList:list, maxSequenceLength:int ,tokenizer:Tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(wordList), maxlen=maxSequenceLength)

def loadTokenizer(tokenizerPath:str='tokenizer_big.pickle'):
    print('[INFO] Load tokenizer.')
    with open(tokenizerPath, 'rb') as p:
        tokenizer = pickle.load(p)
    return tokenizer