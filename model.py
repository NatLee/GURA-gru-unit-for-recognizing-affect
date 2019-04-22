
import pathlib
import numpy as np
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Reshape, Conv1D, Dropout, GlobalMaxPool1D, Conv2D, DepthwiseConv2D, MaxPool2D, TimeDistributed, BatchNormalization, multiply, Flatten, Masking, concatenate, Lambda
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K

from utils import cleanText, getPaddingSequence, saveTrainingImg
from custom import VATModel, QRNN, AttentionDecoder, ElmoEmbeddingLayer



def trainModel(trainingData:tuple, model:Model, modelName:str, datasetName:str, epochs:int=100, validationData:tuple=None, patience:int=10, batchSize:int=1024):
    
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
        hist = model.fit(x=trainingData[0], y=trainingData[1], epochs=epochs, batch_size=batchSize, verbose=1, validation_data=validationData, shuffle=True, callbacks = [earlyStopping, modelCheckpoint])
    else:
        hist = model.fit(x=trainingData[0], y=trainingData[1], epochs=epochs, batch_size=batchSize, verbose=1, validation_split=0.2, shuffle=True, callbacks = [earlyStopping, modelCheckpoint])

    saveTrainingImg(hist, trainingDataImgPath.absolute().as_posix())
    return hist

def testModel(model:Model, maxSequenceLength:int, tokenizer:Tokenizer, testContext:list=['']):
    testContext = ['Tate no Yuusha no Nariagari is one of the best and most wonderful Isekai series of all time. The light novel and the manga is a masterpiece and the beautiful anime adaption in combination with the amazing music make it really majestic.', 'Tate no Yuusha no Nariagari is one of the best and most wonderful Isekai series of all time. Naofumis outfit along with the team members and the other heroes are very stylish and elegant and his relationship with Raphtalia make an excellent couple together, the alluring silhouetes and elegant appearances along with how their relationship links the fates makes the progression of the main character throughout the setting a pure excellence and the overall nature of Tate no Yuusha embraces a wonderful feeling.']
    X_test = getPaddingSequence([cleanText(text) for text in testContext], maxSequenceLength, tokenizer)
    Y_predict = model.predict(x=X_test, batch_size=2048, verbose=1)
    print(Y_predict)
    return Y_predict

'''
def model1(maxSequenceLength:int, embeddingDim:int):
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))
    m = Masking(mask_value=0)(inputs)
    b = Bidirectional(LSTM(20, dropout=0.1, return_sequences=True))(m)
    a = AttentionDecoder(20, 20)(b)
    a = Bidirectional(LSTM(20, dropout=0.1, return_sequences=True))(a)
    q = QRNN(20, return_sequences=True, window_size=3)(a)
    q = QRNN(20, return_sequences=False, window_size=1)(q)
    output = Dense(1, activation='sigmoid')(q)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
'''


def model1(maxSequenceLength:int, embeddingDim:int):
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))
    b = Bidirectional(LSTM(256, dropout=0.1, return_sequences=True))(inputs)
    g = GlobalMaxPool1D()(b)
    d = Dense(256, activation='selu')(g)
    d = Dense(256, activation='selu')(d)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    print(model.summary())
    return model


def modelQNN(maxSequenceLength:int, embeddingDim:int):
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))
    q = QRNN(64, activation='relu', window_size=10, return_sequences=True)(inputs)
    q = QRNN(64, activation='relu', return_sequences=True, window_size=10)(q)
    q = QRNN(64, activation='relu', return_sequences=True, window_size=5)(q)
    q = QRNN(64, activation='relu', return_sequences=False, window_size=1)(q)
    output = Dense(1, activation='sigmoid')(q)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def model2(maxSequenceLength:int, embeddingDim:int):
    # https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))
    b = Bidirectional(LSTM(int(maxSequenceLength/2), dropout=0.2, return_sequences=True))(inputs)
    g = GlobalMaxPool1D()(b)
    d = Dense(20, activation='selu')(g)
    d = Dropout(0.05)(d)
    output = Dense(1, activation='sigmoid')(d)
    model = Model(inputs=inputs, outputs=output)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    print(model.summary())
    return model

def model1_VAT(maxSequenceLength:int, embeddingDim:int):
    inputs = Input(shape=(maxSequenceLength, embeddingDim, ))
    m = Masking(mask_value=0)(inputs)
    b = Bidirectional(LSTM(20, dropout=0.1, return_sequences=True))(m)
    a = AttentionDecoder(20, 20)(b)
    a = Bidirectional(LSTM(20, dropout=0.1, return_sequences=True))(a)
    q = QRNN(20, return_sequences=True, window_size=3)(a)
    q = QRNN(20, return_sequences=False, window_size=1)(q)
    output = Dense(1, activation='sigmoid')(q)
    model = VATModel(inputs, output).setup_vat_loss()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
