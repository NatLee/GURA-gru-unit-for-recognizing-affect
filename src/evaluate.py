import os
import argparse
import logging
import pickle
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

from embedding import EmbeddingPrediction
from utils import cleanText, getPaddingSequence, loadTokenizer, saveAccHist
from dataLoader import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s : %(message)s')

# python -i main.py --imdb --sst2 --mpqa --mr --subj --anime
parser = argparse.ArgumentParser(description='Dataset parse.')
parser.add_argument('--imdb', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--sst2', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--mpqa', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--mr', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--subj', action='store', nargs='?', const=True, default=False, type=bool)
parser.add_argument('--anime', action='store', nargs='?', const=True, default=False, type=bool)
config = parser.parse_args()

EMB_WEIGHT_PATH = 'embeddingLayerWeight.h5'
PRETRAIN_EMB_WEIGHT_PATH = 'wordVector/glove.840B.300d.txt'
USE_PICKLE = True
embeddingDim = 300
maxSequenceLength = 128
epochs = 100
batchSize = 300
modelName = 'modelLstmCnn'
modelFolder = Path('models/' + modelName)
modelFiles = [x for x in modelFolder.glob('**/*.h5') if x.is_file()]
target_names = ['Negative', 'Positive']

tokenizer = loadTokenizer()

logging.info('Load embedding Layer.')
emb = EmbeddingPrediction(modelWeightPath=EMB_WEIGHT_PATH, wordVectorPath=PRETRAIN_EMB_WEIGHT_PATH, tokenizer=tokenizer, embeddingDim=embeddingDim)

for modelFile in modelFiles:
    modelFilePath = modelFile.absolute().as_posix()
    print(modelFilePath)

    if modelFilePath.find('imdb') >= 0:
        #X_vali, Y_vali, imdb_X_train, imdb_Y_train = loadOfficialIMDB(maxSequenceLength, tokenizer, USE_PICKLE)
        continue

    elif modelFilePath.find('mpqa') >= 0:
        #X_vali, Y_vali, mpqa_X, mpqa_Y = loadMPQA(maxSequenceLength, tokenizer, USE_PICKLE)
        continue

    elif modelFilePath.find('mr') >= 0:
        #X_vali, Y_vali, mr_X, mr_Y = loadMR(maxSequenceLength, tokenizer, USE_PICKLE)
        continue

    elif modelFilePath.find('anime') >= 0:
        #X_vali, Y_vali, anime_X, anime_Y = loadAnimePickle(maxSequenceLength, tokenizer, USE_PICKLE)

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

            X = X_pos + X_neg
            Y = Y_pos + Y_neg
            X_vali = getPaddingSequence(X, 128, tokenizer)         
            Y_vali = np.asarray(Y)

        #continue

    elif modelFilePath.find('sst2') >= 0:
        #X_vali, Y_vali, sst2_X_train, sst2_Y_train = loadSST2(maxSequenceLength, tokenizer, USE_PICKLE)
        continue

    else:
        continue

    X_vali_emb = emb.getEmbeddingVector(X_vali)

    m = load_model(modelFilePath, custom_objects={'SeqSelfAttention': SeqSelfAttention})

    y_pred = m.predict(X_vali_emb, verbose=1, batch_size=1024)
    y_pred = [ 1 if y > 0.5 else 0 for y in y_pred ]

    print(accuracy_score(Y_vali, y_pred))
    print(classification_report(Y_vali, y_pred, target_names=target_names))


'''
with open('prediction.pkl', 'wb') as p:
    pickle.dump((Y_vali, y_pred), p)
'''