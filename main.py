import pickle
import os
from embedding import EmbeddingPrediction
from utils import cleanText, getPaddingSequence, loadTokenizer

from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EMB_WEIGHT_PATH = 'embeddingLayerWeight.glove.840B.300d.h5'
PRETRAIN_EMB_WEIGHT_PATH = 'wordVector/glove.840B.300d.txt'
#PRETRAIN_EMB_WEIGHT_PATH = 'wordVector/glove.twitter.27B.200d.txt'
EMB_DIM = 300
maxSequenceLength = 128
epochs = 100
batchSize = 1000

tokenizer = loadTokenizer()

# TODO: 拆出 dataloader 包embedding 寫成 class

print('[INFO] Load and clean the data.')
from dataLoader import *

imdb_X_vali, imdb_Y_vali, imdb_X_train, imdb_Y_train = loadIMDB(maxSequenceLength, tokenizer, True)


sst2_X_vali, sst2_Y_vali, sst2_X_train, sst2_Y_train = loadSST2(maxSequenceLength, tokenizer, True)
mpqa_X, mpqa_Y = loadMPQA(maxSequenceLength, tokenizer, True)
mr_X, mr_Y = loadMR(maxSequenceLength, tokenizer, True)
subj_X, subj_Y = loadSUBJ(maxSequenceLength, tokenizer, True)
anime_X, anime_Y = loadAnimePickle(maxSequenceLength, tokenizer, True)


print('[INFO] Load embedding Layer.')
emb = EmbeddingPrediction(modelWeightPath=EMB_WEIGHT_PATH, wordVectorPath=PRETRAIN_EMB_WEIGHT_PATH, tokenizer=tokenizer, embeddingDim=EMB_DIM, maxSequenceLength=maxSequenceLength)
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


from model import *


m1 = model1(maxSequenceLength=maxSequenceLength, embeddingDim=EMB_DIM)
imdb_m1_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=m1, modelName='model1', datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))


sst2_m1_hist = trainModel(trainingData=(sst2_X_train_emb, sst2_Y_train), model=m1, modelName='model1', datasetName='sst2', epochs=epochs, batchSize=batchSize, validationData=(sst2_X_vali_emb, sst2_Y_vali))
mpqa_m1_hist = trainModel(trainingData=(mpqa_X_emb, mpqa_Y), model=m1, modelName='model1', datasetName='mpqa', epochs=epochs, batchSize=batchSize)
mr_m1_hist = trainModel(trainingData=(mr_X_emb, mr_Y), model=m1, modelName='model1', datasetName='mr', epochs=epochs, batchSize=batchSize)
subj_m1_hist = trainModel(trainingData=(subj_X_emb, subj_Y), model=m1, modelName='model1', datasetName='subj', epochs=epochs, batchSize=batchSize)
anime_m1_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=m1, modelName='model1', datasetName='anime', epochs=epochs, batchSize=batchSize)




'''
mqnn = modelQNN(maxSequenceLength=maxSequenceLength, embeddingDim=EMB_DIM)
imdb_mqnn_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=mqnn, modelName='mqnn', datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))


sst2_mqnn_hist = trainModel(trainingData=(sst2_X_train_emb, sst2_Y_train), model=mqnn, modelName='mqnn', datasetName='sst2', epochs=epochs, batchSize=batchSize, validationData=(sst2_X_vali_emb, sst2_Y_vali))
mpqa_mqnn_hist = trainModel(trainingData=(mpqa_X_emb, mpqa_Y), model=mqnn, modelName='mqnn', datasetName='mpqa', epochs=epochs, batchSize=batchSize)
mr_mqnn_hist = trainModel(trainingData=(mr_X_emb, mr_Y), model=mqnn, modelName='mqnn', datasetName='mr', epochs=epochs, batchSize=batchSize)
subj_mqnn_hist = trainModel(trainingData=(subj_X_emb, subj_Y), model=mqnn, modelName='mqnn', datasetName='subj', epochs=epochs, batchSize=batchSize)
anime_mqnn_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=mqnn, modelName='mqnn', datasetName='anime', epochs=epochs, batchSize=batchSize)



m1_vat = model1_VAT(maxSequenceLength=maxSequenceLength, embeddingDim=EMB_DIM)
imdb_m1_vat_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=m1_vat, modelName='model1_vat', datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))
#anime_m1_vat_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=m1_vat, modelName='model1_vat', datasetName='anime', epochs=epochs, batchSize=batchSize)

m2 = model2(maxSequenceLength=maxSequenceLength, embeddingDim=EMB_DIM)
imdb_m2_hist = trainModel(trainingData=(imdb_X_train_emb, imdb_Y_train), model=m2, modelName='model2', datasetName='imdb', epochs=epochs, batchSize=batchSize, validationData=(imdb_X_vali_emb, imdb_Y_vali))
#anime_m2_hist = trainModel(trainingData=(anime_X_emb, anime_Y), model=m2, modelName='model2', datasetName='anime', epochs=epochs, batchSize=batchSize)
'''