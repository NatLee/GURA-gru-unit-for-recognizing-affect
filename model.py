
import pathlib
import numpy as np
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split

from utils import cleanText, getPaddingSequence, saveTrainingImg
from custom import *


def trainModel(trainingData:tuple, model:Model, modelName:str, datasetName:str, epochs:int=100, validationData:tuple=None, patience:int=5, batchSize:int=1024):
    
    modelsPath = pathlib.Path('models')
    if not modelsPath.exists():
        modelsPath.mkdir()
    folder = modelsPath / modelName
    if not folder.exists():
        folder.mkdir()
    modelPath = folder / (datasetName + '-epoch_{epoch:04d}-acc_{acc:.4f}-valAcc_{val_acc:.4f}.h5')
    modelStructureImg = folder/ 'modelStructurePlot.png'
    trainingDataImgPath = folder / ('{}_{}.png'.format(datasetName, modelName))
    earlyStopping = EarlyStopping(patience = patience)
    modelCheckpoint = ModelCheckpoint(filepath=modelPath.absolute().as_posix(), save_best_only=True, save_weights_only=False, monitor='val_acc', mode='auto')
    plot_model(model, to_file=modelStructureImg.absolute().as_posix(), show_shapes=True, show_layer_names=True)

    if validationData is not None:
        hist = model.fit(x=trainingData[0], y=trainingData[1], epochs=epochs, batch_size=batchSize, verbose=1, validation_data=validationData, shuffle=True, callbacks=[earlyStopping, modelCheckpoint])
    else:
        hist = model.fit(x=trainingData[0], y=trainingData[1], epochs=epochs, batch_size=batchSize, verbose=1, validation_split=0.2, shuffle=True, callbacks=[earlyStopping, modelCheckpoint])

    saveTrainingImg(hist, trainingDataImgPath.absolute().as_posix())
    return hist

def testModel(model:Model, maxSequenceLength:int, tokenizer:Tokenizer, testContext:list=['']):
    #testContext = ['Tate no Yuusha no Nariagari is one of the best and most wonderful Isekai series of all time. The light novel and the manga is a masterpiece and the beautiful anime adaption in combination with the amazing music make it really majestic.', 'Tate no Yuusha no Nariagari is one of the best and most wonderful Isekai series of all time. Naofumis outfit along with the team members and the other heroes are very stylish and elegant and his relationship with Raphtalia make an excellent couple together, the alluring silhouetes and elegant appearances along with how their relationship links the fates makes the progression of the main character throughout the setting a pure excellence and the overall nature of Tate no Yuusha embraces a wonderful feeling.']
    X_test = getPaddingSequence([cleanText(text) for text in testContext], maxSequenceLength, tokenizer)
    Y_predict = model.predict(x=X_test, batch_size=2048, verbose=1)
    print(Y_predict)
    return Y_predict

def modelOurs(maxSequenceLength:int, embeddingDim:int):
    
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))

    b = Bidirectional(GRU(128, return_sequences=True))(inputs)
    b = SeqSelfAttention()(b)
    g = GlobalMaxPool1D()(b)
    d = Dense(64, activation='selu')(g)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


def modelCnn(maxSequenceLength:int, embeddingDim:int):
    
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))

    b = Conv1D(256, 3)(inputs)
    g = GlobalMaxPool1D()(b)
    d = Dense(64, activation='selu')(g)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    return model

def modelLstm(maxSequenceLength:int, embeddingDim:int):
    
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))

    b = LSTM(256, return_sequences=True)(inputs)
    g = GlobalMaxPool1D()(b)
    d = Dense(64, activation='selu')(g)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    return model


def modelLstmCnn(maxSequenceLength:int, embeddingDim:int):
    
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))

    b = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    b = Conv1D(256, 3)(b)
    g = GlobalMaxPool1D()(b)
    d = Dense(64, activation='selu')(g)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

