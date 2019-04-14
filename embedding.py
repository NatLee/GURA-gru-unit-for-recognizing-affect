from keras.models import Model, Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Input
from tqdm import tqdm
import numpy as np

class EmbeddingPrediction(object):

    def __init__(self, modelWeightPath:str=None, wordVectorPath:str=None, tokenizer:Tokenizer=None, embeddingDim:int=300, maxSequenceLength:int=128, trainable:bool=False):
        
        self.__wordVectorPath = wordVectorPath
        self.__modelWeightPath = modelWeightPath

        self.__embeddingType = 'float32'
        self.__embeddingDim = embeddingDim
        
        self.__maxSequenceLength = maxSequenceLength

        self.__tokenizer = tokenizer
        self.__wordIndexSize = len(self.__tokenizer.word_index)+1

        self.__trainable = trainable
        self.__model = self.__buildModel()


    def __buildModel(self):
        model = Sequential()
        if self.__modelWeightPath is not None:
            print('[INFO] Find embedding weight path. Now loading...')
            model.add(Embedding(self.__wordIndexSize, self.__embeddingDim, dtype=self.__embeddingType, input_length=self.__maxSequenceLength, trainable=self.__trainable))
            model.load_weights(self.__modelWeightPath, by_name=True)
        else:
            print('[INFO] Embedding weight path not found. Now Generating...')
            wordVectorMatrix = self.__getWordVectorMatrix()
            model.add(Embedding(self.__wordIndexSize, self.__embeddingDim, dtype=self.__embeddingType, weights=[wordVectorMatrix], input_length=self.__maxSequenceLength, trainable=self.__trainable))
            model = self.__getEmbeddingModel(model)
        return model

    def __getEmbeddingModel(self, model:Model, modelWeightPath:str='embeddingLayerWeight.h5') -> Model:
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        print('[INFO] Saving weight in ./%s' % modelWeightPath)
        model.save_weights(modelWeightPath)
        return model

    def getEmbeddingVector(self, inputX, batchSize:int=2048):
        return self.__model.predict(inputX, batch_size=batchSize, verbose=1)

    def __getWordVectorMatrix(self):

        wordDict = self.__loadWordVector()

        wordVectorMatrix = np.zeros((self.__wordIndexSize, self.__embeddingDim), dtype=self.__embeddingType)
        for word, i in self.__tokenizer.word_index.items():
            if i >= self.__wordIndexSize:
                continue
            wordVector = wordDict.get(str(word))
            if wordVector is not None:
                wordVectorMatrix[i] = wordVector
        return wordVectorMatrix

    def __loadWordVector(self):
        wordDict = dict()
        print('[INFO] Load pretrain word vector weight file.')
        with open(self.__wordVectorPath, 'r', encoding='utf-8') as embFile:
            for line in tqdm(embFile.readlines(), ascii = True):
                line = line[:-1]
                wordVec = line.split()
                if wordVec != '':
                    try:
                        wordDict[str(' '.join(wordVec[:len(wordVec)-self.__embeddingDim]))] = np.asarray(wordVec[len(wordVec)-self.__embeddingDim:len(wordVec)], dtype='float32')
                    except Exception as e:
                        print(e)
                        print(line)
        return wordDict
