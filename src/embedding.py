from keras.models import Model, Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Input
from keras import backend as K
from tqdm import tqdm
import numpy as np
import requests
import zipfile
import pathlib
import os
from loguru import logger

class EmbeddingPrediction(object):

    def __init__(self, modelWeightPath:str=None, wordVectorPath:str=None, tokenizer:Tokenizer=None, embeddingDim:int=300, trainable:bool=False, model_path = '/model'):

        self.model_path = pathlib.Path(model_path)

        self.__wordVectorPath = wordVectorPath
        self.__modelWeightPath = modelWeightPath

        self.__embeddingType = 'float32'
        
        self.__embeddingDim = embeddingDim

        self.__tokenizer = tokenizer
        self.__wordIndexSize = len(self.__tokenizer.word_index)+1

        self.__trainable = trainable
        self.__model = self.__buildModel()


    def __buildModel(self):
        K.set_floatx(self.__embeddingType)

        model = Sequential()
        
        if self.__modelWeightPath is None:
            model_weight_path = self.model_path / 'embeddingLayerWeight.h5'
            if model_weight_path.exists():
                self.__modelWeightPath = model_weight_path
        
        if self.__modelWeightPath is not None:
            logger.info('Find embedding weight path. Now loading...')
            model.add(Embedding(input_dim=self.__wordIndexSize, output_dim=self.__embeddingDim, dtype=self.__embeddingType, trainable=self.__trainable))
            model.load_weights(self.__modelWeightPath, by_name=True)
        else:
            logger.warning('Embedding weight path not found. Now Generating...')
            wordVectorMatrix = self.__getWordVectorMatrix()
            model.add(Embedding(input_dim=self.__wordIndexSize, output_dim=self.__embeddingDim, dtype=self.__embeddingType, weights=[wordVectorMatrix], trainable=self.__trainable))
            model = self.__getEmbeddingModel(model)
        return model

    def __getEmbeddingModel(self, model:Model, modelWeightPath:str='/model/embeddingLayerWeight.h5') -> Model:
        model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
        print(model.summary())
        logger.info('Saving weight in ./%s' % modelWeightPath)
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

        gloveZipFileDownloadPath = self.model_path / 'glove.840B.300d.zip'
        wordVecFolderPath = self.model_path / 'wordVector'

        # ====================
        # check path exists
        if self.__wordVectorPath is None:
            vector_path = wordVecFolderPath / 'glove.840B.300d.txt'
            if vector_path.exists():
                self.__wordVectorPath = vector_path
        # ====================

        if self.__wordVectorPath is None:
            print('[WARNING] Word vector file path not found.')

            gloveUrl = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

            print(f'[WARNING] Downloading from `{gloveUrl}` into `model` folder')

            if not wordVecFolderPath.is_dir():
                wordVecFolderPath.mkdir()
            
            self.__downloadFile(gloveUrl, gloveZipFileDownloadPath)
            unzipFiles = self.__unzip(gloveZipFileDownloadPath, wordVecFolderPath)
            os.remove(gloveZipFileDownloadPath)
            for unzipFile in unzipFiles:
                if unzipFile.find('glove') >= 0:
                    self.__wordVectorPath = (wordVecFolderPath / unzipFile).absolute().as_posix()
                    break

        logger.info('Load pre-trained word vector weight file.')
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

    def __downloadFile(self, url:str, filePath:pathlib.Path):
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            logger.info('Downloading file... [{}]'.format(filePath.name))
            r.raise_for_status()
            with open(filePath.absolute().as_posix(), 'wb') as f:
                pbar = tqdm(unit='B', total=int(r.headers['Content-Length']), ascii=True)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        pbar.update (len(chunk))
                        _ = f.write(chunk)
        return filePath

    def __unzip(self, fileName:str, folderPath:pathlib.Path):
        with open(fileName, 'rb') as f:
            logger.info(f'Now unzipping file... [{fileName}]')
            z = zipfile.ZipFile(f)
            for name in z.namelist():
                z.extract(name, folderPath.absolute().as_posix())
            return z.namelist()