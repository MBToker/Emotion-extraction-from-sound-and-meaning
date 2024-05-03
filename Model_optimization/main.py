import math
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scikeras.wrappers import KerasClassifier 
from tensorflow.keras.constraints import max_norm
from scipy.stats import randint
from tensorflow.keras.callbacks import TensorBoard
import keras
import pickle
import time
import os
import datetime
import ast

#Extracts the sound features
def extract_sound_feature(audio_file, max_length=100, n_mfcc=20, n_fft=512):
    dur = len(audio_file)
    audio, sr=librosa.load(audio_file, duration=math.floor(float(dur)), offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


#Importing sound files
def import_sound_files(table_df, sound_df, folder_name):
    for i in range(len(table_df)):
       try:
          file_name = f"{folder_name}/{table_df['fileName'][i]}.wav"
          features = extract_sound_feature(file_name, 150, 20, 512)
        # Add the remaining code inside the try block
       except Exception as e:
          print(f"Error occurred at index {i}: {e}")

       temp_df=pd.DataFrame({
           'fileName': [table_df['fileName'][i]],
           'soundFile': [features]
           }) 
       sound_df=pd.concat([sound_df, temp_df], ignore_index=True)
    return sound_df


def lstm_training_preprocess(table_df):
    le_x = LabelEncoder()
    enc_y = OneHotEncoder()
    table_df['Sentiment'] = le_x.fit_transform(table_df['Sentiment']).astype(float)

    #Adding "Sentiment" and "soundFile" column together
    
    table_df['soundFile'] = np.array(table_df['soundFile'])
    table_df['soundFile'] = table_df['soundFile'].apply(lambda x: np.array(ast.literal_eval(x)))

    for i in range(len(table_df['soundFile'])):
        table_df['soundFile'][i] = np.append(table_df['soundFile'][i], table_df['Sentiment'][i])
    
    x=table_df['soundFile']
    y=enc_y.fit_transform(table_df[['Emotion']])

    x = [i for i in x]
    x = np.array(x)
    x = np.expand_dims(x, -1)
    y = y.toarray()

    return x,y


def define_model(neurons=10, dropout_rate=0.0, weight_constraints=0, optimizers='Adam'):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, kernel_initializer= 'he_uniform',
                   input_shape=(21,1), kernel_constraint=max_norm(weight_constraints)))
    model.add(LSTM(neurons, return_sequences=True, input_shape=(21,1), activation='relu', 
                   kernel_constraint=max_norm(weight_constraints), kernel_initializer= 'he_uniform'))
    model.add(Dense(neurons, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7, kernel_initializer='he_uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers,
                  metrics=['accuracy'])
    return model


# Auto hyperparameter tuning
def opt_gridcv(train_df, val_df):
    x, y= lstm_training_preprocess(train_df)
    val_x, val_y= lstm_training_preprocess(val_df)

    model=KerasClassifier(build_fn=define_model, verbose=1)
    model__dropout_rate = [0.2, 0.4]
    model__weight_constraints = [1, 2, 3]
    model__neurons = [64, 128, 256]
    model__optimizers = ['SGD', 'Nadam', 'SGD', 'RMSprop']
    #activation = ['relu', 'tanh', 'sigmoid']
    batch_size=[128, 256, 512]
    epochs = [75, 89]

    #neurons, dropout_rate, weight_constraints, optimizer, activation
    param_grid = dict(model__neurons=model__neurons, model__dropout_rate=model__dropout_rate, 
                      model__weight_constraints=model__weight_constraints, model__optimizers=model__optimizers,
                      batch_size=batch_size, epochs=epochs)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=6, cv=3)
    grid_result = grid.fit(x, y, validation_data=(val_x, val_y))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Mean = %f (std=%f) with %r" % (mean,stdev, param))

    print("Finished")


def opt_randomcv(train_df, val_df):
    x, y = lstm_training_preprocess(train_df)
    val_x, val_y= lstm_training_preprocess(val_df)

    model = KerasClassifier(build_fn=define_model, verbose=1)
    param_dist = {
        'model__neurons': randint(64, 256),
        'model__dropout_rate': [0.2, 0.4],
        'model__weight_constraints': [1, 2, 3],
        'model__optimizers': ['SGD', 'Nadam', 'RMSprop', 'Adam'],
        'batch_size': [128, 256, 512],
        'epochs': [75, 89]
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=4)
    random_search_result = random_search.fit(x, y, validation_data=(val_x, val_y))

    print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
    means = random_search_result.cv_results_['mean_test_score']
    stds = random_search_result.cv_results_['std_test_score']
    params = random_search_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Mean = %f (std=%f) with %r" % (mean, stdev, param))

    print("Finished")


def create_dataframe(temp_df):
    table_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])

    for i in range(len(temp_df)): 
          new_fileName=temp_df.loc[i]['fileName']
          new_Emotion=temp_df.loc[i]['Emotion']
          new_Sentiment=temp_df.loc[i]['Sentiment']
          new_Content=temp_df.loc[i]["Utterance"].lower()

          new_line=pd.DataFrame({
             'fileName': [new_fileName],
             'Emotion': [new_Emotion],
             'Content': [new_Content],
             'Sentiment': [new_Sentiment]
          })

          table_df=pd.concat([table_df,new_line], ignore_index=True)

    return table_df


def dataframe_preprocessing(data_folder, unprocessed_df):
    sound_df=pd.DataFrame(columns=['fileName', 'soundFile'])
    document_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])
    document_df=create_dataframe(unprocessed_df)
    unprocessed_df.drop(unprocessed_df.index, inplace=True)

    sound_df=import_sound_files(document_df, sound_df, data_folder)
    document_df=pd.concat([document_df, sound_df], axis=1, join='outer')
    sound_df.drop(sound_df.index, inplace=True)
    return document_df


# Find optimum model layer count and neuron size with TensorBoard
def find_optimum_model(train_df, val_df):
    dense_layers = [2, 3]
    layer_sizes = [64, 128, 256]
    lstm_layers = [1, 2, 3]

    train_x, train_y = lstm_training_preprocess(train_df)
    val_x, val_y= lstm_training_preprocess(val_df)
    count=0
    
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for lstm_layer in lstm_layers:
                name = "{}-lstm-{}-nodes-{}-dense-{}".format(lstm_layer,layer_size,dense_layer,int(time.time()))
                tensorboard_callback = keras.callbacks.TensorBoard(
                    log_dir="logs/{}".format(name), histogram_freq=1,
                )
                model = Sequential()

                for l in range(lstm_layer):
                    model.add(LSTM(layer_size, return_sequences=True, input_shape=(21,1), activation='relu'))

                model.add(LSTM(layer_size, return_sequences=False, input_shape=(21,1), activation='relu'))
                    
                for i in range(dense_layer):
                    model.add(Dense(layer_size, activation='relu'))
                    model.add(Dropout(0.2))

                model.add(Dense(7, activation="softmax"))

                model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
                model.fit(train_x, train_y, batch_size=64, epochs=64, validation_data=(val_x, val_y), callbacks=[tensorboard_callback], shuffle=True)


# Data and excel paths
"""train_data_folder = 'C:/Emotion_Recognition_From_Speech/Kodlar/Emotion_recog_from_sound/dataset/training'
val_data_folder = 'C:/Emotion_Recognition_From_Speech/Kodlar/Emotion_recog_from_sound/dataset/val'
unprocessed_train_df=pd.read_excel('train_excel.xlsx')
unprocessed_val_df=pd.read_excel('val_excel.xlsx')

# Defining variables
C:\Emotion_Recognition_From_Speech\Kodlar\Emotion_recog_from_sound\dataset\val
train_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])
val_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])

# Processing the data
train_df=dataframe_preprocessing(train_data_folder, unprocessed_train_df)
val_df=dataframe_preprocessing(val_data_folder, unprocessed_val_df)

# Saving the processed data to make process faster
formatter = {'float_kind':lambda x: "%.6f," % x}
train_df['soundFile'] = train_df['soundFile'].apply(lambda x: np.array2string(x, formatter=formatter))
val_df['soundFile'] = val_df['soundFile'].apply(lambda x: np.array2string(x, formatter=formatter))
train_df.to_excel('train_df1.xlsx', index=False)
val_df.to_excel('val_df1.xlsx', index=False)"""

# Load the saved processed data
train_df=pd.read_excel('train_df1.xlsx')
val_df=pd.read_excel('val_df1.xlsx')

# Optimization choices
find_optimum_model(train_df, val_df)
#opt_gridcv(train_df, val_df)
#opt_randomcv(train_df, val_df)
