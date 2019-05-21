import re
import pickle
import json

from pathlib import Path
from unidecode import unidecode
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import History

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
    #ax2.set_ylim(0,1)
    #ax2.set_yticks(np.arange(0, 1, 0.1))
    ax2.set_ylabel('Acc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax2.legend()

    plt.savefig(savePath)
    return True


def cleanText(text):

    text = text.lower()
    #cleanHtmlTag = re.compile('<.*?>')
    #text = re.sub(cleanHtmlTag, '', text)
    
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    
    stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards']
    stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along']
    stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
    stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
    stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
    stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
    stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
    stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
    stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
    stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
    stopwords += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
    stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
    stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
    stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
    stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
    stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
    stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
    stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
    stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
    stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
    stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
    stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
    stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
    stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
    stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
    stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
    stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
    stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
    stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
    stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
    stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
    stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
    stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
    stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
    stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
    stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
    stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
    stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
    stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
    stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
    stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
    stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
    stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
    stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
    stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
    stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
    stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
    stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
    stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
    stopwords += ['within', 'without', 'would', 'yet', 'you', 'your']
    stopwords += ['yours', 'yourself', 'yourselves']

    def removeStopwords(wordlist, stopwords):
        return [w for w in wordlist if w not in stopwords]

    text = ' '.join(removeStopwords(text.split(), stopwords))
    
    text = text.strip()

    return text

def getPaddingSequence(wordList:list, maxSequenceLength:int ,tokenizer:Tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(wordList), maxlen=maxSequenceLength)

def loadTokenizer(tokenizerPath:str='tokenizer_big.pickle'):
    print('[INFO] Load tokenizer.')
    with open(tokenizerPath, 'rb') as p:
        tokenizer = pickle.load(p)
    return tokenizer


def saveAccHist(currentHist:History, datasetName:str, accHistoryFilePath:str='accHistory.json'):
    accHistoryFilePath = Path(accHistoryFilePath)
    accHistoryFilePathString = accHistoryFilePath.absolute().as_posix()
    if not accHistoryFilePath.exists():
        with open(accHistoryFilePathString, 'w') as f:
            json.dump({}, f)
    with open(accHistoryFilePathString, 'r') as f:
        accHistory = json.load(f)
    accHistory[datasetName] = sorted(currentHist.history['val_acc'])[-1]
    with open(accHistoryFilePathString, 'w') as f:
        json.dump(accHistory, f)


def tsnePlot(embedVecs, polarity):

    labels = []
    tokens = []

    for i, sentenceVec in enumerate(embedVecs):
        tokens.append(sentenceVec)
        labels.append(polarity)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
