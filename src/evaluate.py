import os
import argparse
import pickle

from loguru import logger

from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

from embedding import EmbeddingPrediction
from utils import cleanText, getPaddingSequence, loadTokenizer, saveAccHist
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

USE_PICKLE = True
embeddingDim = 300
maxSequenceLength = 128

modelName = 'modelOurs'

modelFolder = Path('/model') / modelName
modelFiles = [x for x in modelFolder.glob('**/*.h5') if x.is_file()]
target_names = ['Negative', 'Positive']

tokenizer = loadTokenizer(tokenizerPath='/model/big-tokenizer.pkl')

logger.info('Load embedding Layer.')
emb = EmbeddingPrediction(tokenizer=tokenizer, embeddingDim=embeddingDim)

def predict(x_validation, y_validation, model_file_path):
    valid_embedding = emb.getEmbeddingVector(x_validation)

    m = load_model(model_file_path, custom_objects={'SeqSelfAttention': SeqSelfAttention})

    y_pred = m.predict(valid_embedding, verbose=1, batch_size=256)
    y_pred = [ 1 if y > 0.5 else 0 for y in y_pred ]

    logger.info(accuracy_score(y_validation, y_pred))
    logger.info(classification_report(y_validation, y_pred, target_names=target_names))

for modelFile in modelFiles:
    modelFilePath = modelFile.absolute().as_posix()
    logger.info(f'Model file path -> {modelFilePath}')

    if modelFilePath.find('imdb') >= 0:
        X_vali, Y_vali, imdb_X_train, imdb_Y_train = loadOfficialIMDB(maxSequenceLength, tokenizer, USE_PICKLE)
        predict(X_vali, Y_vali, modelFilePath)

    if modelFilePath.find('mpqa') >= 0:
        X_vali, Y_vali, mpqa_X, mpqa_Y = loadMPQA(maxSequenceLength, tokenizer, USE_PICKLE)
        predict(X_vali, Y_vali, modelFilePath)

    if modelFilePath.find('mr') >= 0:
        X_vali, Y_vali, mr_X, mr_Y = loadMR(maxSequenceLength, tokenizer, USE_PICKLE)
        predict(X_vali, Y_vali, modelFilePath)

    if modelFilePath.find('anime') >= 0:
        X_vali, Y_vali, anime_X, anime_Y = loadAnimePickle(maxSequenceLength, tokenizer, USE_PICKLE)
        datasetName = '/data/myanimelist-sts.pkl'
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

            X = X_pos + X_neg
            Y = Y_pos + Y_neg
            X_vali = getPaddingSequence(X, 128, tokenizer)         
            Y_vali = np.asarray(Y)

        predict(X_vali, Y_vali, modelFilePath)

    if modelFilePath.find('sst2') >= 0:
        X_vali, Y_vali, sst2_X_train, sst2_Y_train = loadSST2(maxSequenceLength, tokenizer, USE_PICKLE)
        predict(X_vali, Y_vali, modelFilePath)

'''
with open('prediction.pkl', 'wb') as p:
    pickle.dump((Y_vali, y_pred), p)
'''