import pickle
import os
from pathlib import Path

from embedding import EmbeddingPrediction
from utils import cleanText, getPaddingSequence, loadTokenizer
from dataLoader import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EMB_WEIGHT_PATH = 'embeddingLayerWeight.h5'
PRETRAIN_EMB_WEIGHT_PATH = 'wordVector/glove.840B.300d.txt'
USE_PICKLE = False
embeddingDim = 300
maxSequenceLength = 128
epochs = 100
batchSize = 500

tokenizer = loadTokenizer()


imdb_X_vali, imdb_Y_vali, imdb_X_train, imdb_Y_train = loadOfficialIMDB(maxSequenceLength, tokenizer, USE_PICKLE)


sst2_X_vali, sst2_Y_vali, sst2_X_train, sst2_Y_train = loadSST2(maxSequenceLength, tokenizer, USE_PICKLE)
mpqa_X, mpqa_Y = loadMPQA(maxSequenceLength, tokenizer, USE_PICKLE)
mr_X, mr_Y = loadMR(maxSequenceLength, tokenizer, USE_PICKLE)
subj_X, subj_Y = loadSUBJ(maxSequenceLength, tokenizer, USE_PICKLE)
anime_X, anime_Y = loadAnimePickle(maxSequenceLength, tokenizer, USE_PICKLE)



print('[INFO] Load embedding Layer.')
emb = EmbeddingPrediction(modelWeightPath=EMB_WEIGHT_PATH, wordVectorPath=PRETRAIN_EMB_WEIGHT_PATH, tokenizer=tokenizer, embeddingDim=embeddingDim)
#emb = EmbeddingPrediction(wordVectorPath=PRETRAIN_EMB_WEIGHT_PATH, tokenizer=tokenizer, embeddingDim=embeddingDim)
#emb = EmbeddingPrediction(tokenizer=tokenizer, embeddingDim=embeddingDim)
print('[INFO] Preprocess the embedding data.')
imdb_X_train_emb = emb.getEmbeddingVector(imdb_X_train)
imdb_X_vali_emb = emb.getEmbeddingVector(imdb_X_vali)


sst2_X_train_emb = emb.getEmbeddingVector(sst2_X_train)
sst2_X_vali_emb = emb.getEmbeddingVector(sst2_X_vali)
mpqa_X_emb = emb.getEmbeddingVector(mpqa_X)
mr_X_emb = emb.getEmbeddingVector(mr_X)
subj_X_emb = emb.getEmbeddingVector(subj_X)
anime_X_emb = emb.getEmbeddingVector(anime_X)


print('[INFO] Build Model.')



m1 = model1(maxSequenceLength=maxSequenceLength, embeddingDim=embeddingDim)
imdb_m1_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=m1, modelName='model1', datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))


sst2_m1_hist = trainModel(trainingData=(sst2_X_train_emb, sst2_Y_train), model=m1, modelName='model1', datasetName='sst2', epochs=epochs, batchSize=batchSize, validationData=(sst2_X_vali_emb, sst2_Y_vali))
mpqa_m1_hist = trainModel(trainingData=(mpqa_X_emb, mpqa_Y), model=m1, modelName='model1', datasetName='mpqa', epochs=epochs, batchSize=batchSize)
mr_m1_hist = trainModel(trainingData=(mr_X_emb, mr_Y), model=m1, modelName='model1', datasetName='mr', epochs=epochs, batchSize=batchSize)
subj_m1_hist = trainModel(trainingData=(subj_X_emb, subj_Y), model=m1, modelName='model1', datasetName='subj', epochs=epochs, batchSize=batchSize)
anime_m1_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=m1, modelName='model1', datasetName='anime', epochs=epochs, batchSize=batchSize)

