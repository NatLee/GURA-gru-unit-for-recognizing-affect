import os
import argparse

from loguru import logger

from embedding import EmbeddingPrediction
from utils import cleanText, getPaddingSequence, loadTokenizer, saveAccHist, OnlyTopValAcc
from dataLoader import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# python -i main.py --imdb --sst2 --mpqa --mr --subj --anime
parser = argparse.ArgumentParser(description='Dataset parse.')
parser.add_argument('--imdb', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--sst2', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--mpqa', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--mr', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--subj', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--anime', action='store', nargs='?', const=True, default=False, type=bool)
config = parser.parse_args()

USE_PICKLE = False
embeddingDim = 300
maxSequenceLength = 128
epochs = 5
batchSize = 128

tokenizer = loadTokenizer(tokenizerPath='/model/big-tokenizer.pkl')

logger.info('Load embedding Layer.')
emb = EmbeddingPrediction(tokenizer=tokenizer, embeddingDim=embeddingDim)

logger.info('Build Model.')

modelName = 'modelOurs'
m1 = modelOurs(maxSequenceLength=maxSequenceLength, embeddingDim=embeddingDim)
#m1 = modelCnn(maxSequenceLength=maxSequenceLength, embeddingDim=embeddingDim)


if config.imdb:
    imdb_X_vali, imdb_Y_vali, imdb_X_train, imdb_Y_train = loadOfficialIMDB(maxSequenceLength, tokenizer, USE_PICKLE)
    imdb_X_train_emb = emb.getEmbeddingVector(imdb_X_train)
    imdb_X_vali_emb = emb.getEmbeddingVector(imdb_X_vali)
    logger.info('Train IMDB')
    imdb_m1_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=m1, modelName=modelName, datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))
    saveAccHist(imdb_m1_hist, 'imdb')

if config.sst2:
    sst2_X_vali, sst2_Y_vali, sst2_X_train, sst2_Y_train = loadSST2(maxSequenceLength, tokenizer, USE_PICKLE)
    sst2_X_train_emb = emb.getEmbeddingVector(sst2_X_train)
    sst2_X_vali_emb = emb.getEmbeddingVector(sst2_X_vali)
    logger.info('Train SST2')
    sst2_m1_hist = trainModel(trainingData=(sst2_X_train_emb, sst2_Y_train), model=m1, modelName=modelName, datasetName='sst2', epochs=epochs, batchSize=batchSize, validationData=(sst2_X_vali_emb, sst2_Y_vali))
    saveAccHist(sst2_m1_hist, 'sst2')

if config.mpqa:
    mpqa_X_vali, mpqa_Y_vali, mpqa_X, mpqa_Y = loadMPQA(maxSequenceLength, tokenizer, USE_PICKLE)
    mpqa_X_emb = emb.getEmbeddingVector(mpqa_X)
    mpqa_X_vali_emb = emb.getEmbeddingVector(mpqa_X_vali)
    logger.info('Train MPQA')
    mpqa_m1_hist = trainModel(trainingData=(mpqa_X_emb, mpqa_Y), model=m1, modelName=modelName, datasetName='mpqa', epochs=epochs, batchSize=batchSize, validationData=(mpqa_X_vali_emb, mpqa_Y_vali))
    saveAccHist(mpqa_m1_hist, 'mpqa')

if config.mr:
    mr_X_vali, mr_Y_vali, mr_X, mr_Y = loadMR(maxSequenceLength, tokenizer, USE_PICKLE)
    mr_X_emb = emb.getEmbeddingVector(mr_X)
    mr_X_vali_emb = emb.getEmbeddingVector(mr_X_vali)
    logger.info('Train MR')
    mr_m1_hist = trainModel(trainingData=(mr_X_emb, mr_Y), model=m1, modelName=modelName, datasetName='mr', epochs=epochs, batchSize=batchSize, validationData=(mr_X_vali_emb, mr_Y_vali))
    saveAccHist(mr_m1_hist, 'mr')

if config.subj:
    subj_X_vali, subj_Y_vali, subj_X, subj_Y = loadSUBJ(maxSequenceLength, tokenizer, USE_PICKLE)
    subj_X_emb = emb.getEmbeddingVector(subj_X)
    subj_X_vali_emb = emb.getEmbeddingVector(subj_X_vali)
    logger.info('Train SUBJ')
    subj_m1_hist = trainModel(trainingData=(subj_X_emb, subj_Y), model=m1, modelName=modelName, datasetName='subj', epochs=epochs, batchSize=batchSize, validationData=(subj_X_vali_emb, subj_Y_vali))
    saveAccHist(subj_m1_hist, 'subj')

if config.anime:
    anime_X_vali, anime_Y_vali, anime_X, anime_Y = loadAnimePickle(maxSequenceLength, tokenizer, USE_PICKLE)
    anime_X_emb = emb.getEmbeddingVector(anime_X)
    anime_X_vali_emb = emb.getEmbeddingVector(anime_X_vali)
    logger.info('Train ANIME')
    anime_m1_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=m1, modelName=modelName, datasetName='anime', epochs=epochs, batchSize=batchSize, validationData=(anime_X_vali_emb, anime_Y_vali))
    saveAccHist(anime_m1_hist, 'anime')

modelFolder = Path('/model') / modelName
OnlyTopValAcc(modelFolder)

