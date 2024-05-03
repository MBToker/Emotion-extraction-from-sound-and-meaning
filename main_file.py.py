import math
import whisper
import soundfile as sf
import os
import numpy as np
import pandas as pd
import unicodedata as ud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from sklearn.model_selection import GridSearchCV
from keras.constraints import max_norm
from scikeras.wrappers import KerasClassifier # Takes model as function
from matplotlib.widgets import Button
import ast
import seaborn as sns
import pyaudio
import wave
from matplotlib.font_manager import FontProperties
import sounddevice as sd
import threading
import tkinter as tk
from tkinter import filedialog
import tempfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk


#-----FUNCTIONS-----#
latin_letters= {}
def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    return all(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha())


def delete_nonlatin_rows(temp_df):
    for i in range(len(temp_df)):
       if not only_roman_chars(temp_df['Content'][i]):
          temp_df=temp_df.drop(i, inplace=True)


#Preprocessing for NLTK process
def cleaning_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words=set(stopwords.words('english'))
    lemmatizer=WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE) 
    text = text.lower() 
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [word for word in text if not word in stop_words] 
    text = " ".join(text)
    return text


#NLTK training
def text_analysis_training(training_df, test_df):
    for i in training_df.index:
        training_df['Content'][i]=cleaning_text(training_df['Content'][i])
    
    #Vectorizing
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(training_df['Content'])  # Fit only on training data for same data features
    train_data_features = vectorizer.transform(training_df['Content'])
    test_data_features = vectorizer.transform(test_df['Content'])  # Transform both train and test data

    #Splitting the dataset
    le = LabelEncoder()
    x_train=pd.DataFrame(train_data_features.toarray())
    y_train=pd.DataFrame(training_df['Sentiment'])
    y_train=le.fit_transform(y_train)

    x_test=pd.DataFrame(test_data_features.toarray())
    y_test=pd.DataFrame(test_df['Sentiment'])
    y_test=le.fit_transform(y_test)
    
    #Label encoder's real values
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    #Training XGBoost Classifier
    model= xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.01)
    model.fit(x_train, y_train)
    predictions=model.predict(x_test)
    print(classification_report(y_test, predictions))
    return model, vectorizer


#Used to make predictions with NLTK model
def text_analysis_prediction_maker(temp_model, independant_vals_df):
    predictions=model.predict(independant_vals_df['Content'])
    return predictions


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
    
    # Main'deki altı satırla beraber bunu da sil (işlem hızlandırıcı)
    table_df['soundFile'] = table_df['soundFile'].apply(lambda x: np.array(ast.literal_eval(x)))

    for i in range(len(table_df['soundFile'])):
        table_df['soundFile'][i] = np.append(table_df['soundFile'][i], table_df['Sentiment'][i])
    
    x=table_df['soundFile']
    y=enc_y.fit_transform(table_df[['Emotion']])

    x = [i for i in x]
    x = np.array(x)
    x = np.expand_dims(x, -1)
    y = y.toarray()

    return x, y, enc_y


#Training LSTM model
def lstm_model_training(train_df, test_df, val_df, epochs=64):
    x_train, y_train, train_enc = lstm_training_preprocess(train_df)
    x_test, y_test, test_enc = lstm_training_preprocess(test_df)
    x_val, y_val, val_enc = lstm_training_preprocess(val_df)
    

    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(256, activation='relu', return_sequences=False, input_shape=(x_train.shape[1],1))) # If return sequences, it returns every timestep of input data
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_val, y_val), shuffle=True)
    loss, accuracy=model.evaluate(x_test, y_test, verbose=2)

    # Metrics
    prediction=model.predict(x_test)
    test_decoded = test_enc.inverse_transform(y_test)
    prediction_decoded = test_enc.inverse_transform(prediction)
    classes = np.unique(test_decoded).tolist()

    prediction=np.argmax(prediction, axis=1)
    y_test_new=np.argmax(y_test, axis=1)

    scores= {
        "accuracy" : accuracy*100,
        "loss" : loss*100,
        "precision" : precision_score(y_test_new, prediction, average='weighted')*100,
        "recall" : recall_score(y_test_new, prediction, average='weighted')*100,
        "f1" : f1_score(y_test_new, prediction, average='weighted')*100,
    }

    train_graph_values = {'history': history, 'scores': scores, 'epochs': epochs, 'test_decoded': test_decoded, 'prediction_decoded': prediction_decoded, 'classes': classes}
    with open('models/train_graph_values.pkl', 'wb') as f:
       pickle.dump(train_graph_values, f)

    plot_multiple_tabs(history, scores, epochs, test_decoded, prediction_decoded, classes)
    return model, test_enc


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


def plot_multiple_tabs(model_1, scores, epochs, test, prediction, classes):
    fig = plt.figure(figsize=(10, 15))
    epochs = list(range(epochs))
    
    # Button events
    def on_tab1_clicked(event):
       tab1_axes.set_visible(True)
       tab2_axes.set_visible(False)
       tab3_axes.set_visible(False)
       tab4_axes.set_visible(False)
       fig.canvas.draw()

    def on_tab2_clicked(event):
       tab1_axes.set_visible(False)
       tab2_axes.set_visible(True)
       tab3_axes.set_visible(False)
       tab4_axes.set_visible(False)
       fig.canvas.draw()

    def on_tab3_clicked(event):
       tab1_axes.set_visible(False)
       tab2_axes.set_visible(False)
       tab3_axes.set_visible(True)
       tab4_axes.set_visible(False)      
       fig.canvas.draw()

    def on_tab4_clicked(event):
       tab1_axes.set_visible(False)
       tab2_axes.set_visible(False)
       tab3_axes.set_visible(False)
       tab4_axes.set_visible(True)
       fig.canvas.draw()

    # First tab
    tab1_axes = fig.add_subplot(111) # 1x1x1 grid
    acc = model_1.history['accuracy']
    val_acc = model_1.history['val_accuracy']
    tab1_axes.set_title("Accuracy by epochs")
    tab1_axes.plot(epochs, acc, label='train acc.')
    tab1_axes.plot(epochs, val_acc, label='validation acc.')
    tab1_axes.set_xlabel('epochs')
    tab1_axes.set_ylabel('accuracy')
    tab1_axes.legend()

    # Second tab
    tab2_axes = fig.add_subplot(111) # 1x1x1 grid
    tab2_axes.set_visible(False)
    loss = model_1.history['loss']
    val_loss = model_1.history['val_loss']
    tab2_axes.set_title("Loss by epochs")
    tab2_axes.plot(epochs, loss, label='train loss')
    tab2_axes.plot(epochs, val_loss, label='validation loss')
    tab2_axes.set_xlabel('epochs')
    tab2_axes.set_ylabel('loss')
    tab2_axes.legend()

    # Third tab
    tab3_axes = fig.add_subplot(111)
    tab3_axes.set_visible(False)
    labels = list(scores.keys())
    percentages = list(scores.values())
    bars = tab3_axes.bar(labels, percentages)  # Adjust bar width as needed
    tab3_axes.set_ylabel('Percentage')
    tab3_axes.set_title('Scores of the model')
    tab3_axes.set_ylim(0, 100)  # Adjust as needed
    tab3_axes.set_xlabel('Categories')  # Labeling x-axis
    tab3_axes.set_ylabel('Scores')  # Labeling y-axis

    # Adding values on top of each bar
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        tab3_axes.text(bar.get_x() + bar.get_width() / 2, height, '%.2f%%' % percentage, ha='center', va='bottom')

    # Forth tab
    tab4_axes = fig.add_subplot(111)
    tab4_axes.set_visible(False)
    conf_matrix = confusion_matrix(test, prediction)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", ax=tab4_axes, xticklabels=classes, yticklabels=classes, cbar=False)
    tab4_axes.set_xlabel('Prediction')
    tab4_axes.set_ylabel('Actual')
    tab4_axes.set_title('Confusion Matrix')

    # left, bottom, width, height
    ax_tab1 = plt.axes([0.1, 0.005, 0.1, 0.05])
    button_tab1 = Button(ax_tab1, 'Tab 1')
    button_tab1.on_clicked(on_tab1_clicked)

    ax_tab2 = plt.axes([0.3, 0.005, 0.1, 0.05])
    button_tab2 = Button(ax_tab2, 'Tab 2')
    button_tab2.on_clicked(on_tab2_clicked)
    
    ax_tab3 = plt.axes([0.5, 0.005, 0.1, 0.05])
    button_tab3 = Button(ax_tab3, 'Tab 3')
    button_tab3.on_clicked(on_tab3_clicked)

    ax_tab4 = plt.axes([0.7, 0.005, 0.1, 0.05])
    button_tab4 = Button(ax_tab4, 'Tab 4')
    button_tab4.on_clicked(on_tab4_clicked)

    plt.show()
    return fig


#Convert sound to text
def audio_transcribe(folder_name, transcribed_texts, name_list):
    for file in os.listdir(folder_name):
        name = file[:-4]
        name_list.append(name)
        file_path = f"{folder_name}/{file}"
        #audio_data, sample_rate = sf.read(file_path) # sample_rate aşağıdaki fonksiyonda kullanılabilir (şuan hata veriyor)
        result = model.transcribe(file_path)
        transcribed_texts.append(result["text"])


# Functions for user uploaded data
def create_df_for_uploaded_data(transcribed_texts, name_list):
    table_df=pd.DataFrame(columns=['fileName','Content','Sentiment','Emotion'])
    for i in range(len(transcribed_texts)): 
          new_fileName=name_list[i]
          new_Sentiment=""
          new_Content=transcribed_texts[i].lower()
          new_Emotion=""

          new_line=pd.DataFrame({
             'fileName': [new_fileName],
             'Content': [new_Content],
             'Sentiment': [new_Sentiment],
             'Emotion' : [new_Emotion]
          })
          table_df=pd.concat([table_df,new_line], ignore_index=True)
    return table_df


def user_upload_files(user_sound_file_folder):
    transcribed_texts = []
    name_list = []
    model = whisper.load_model("base.en")
    audio_transcribe(user_sound_file_folder, transcribed_texts, name_list)
    uploaded_df=pd.DataFrame(columns=['fileName','Content','Sentiment'])
    uploaded_df=create_df_for_uploaded_data(transcribed_texts, name_list)
    return uploaded_df, name_list


def nltk_user_data_test(nltk_model_file, nltk_vectorizer_file, uploaded_df):
    nltk_model = pickle.load(open(nltk_model_file, "rb")) # Load command
    vectorizer = pickle.load(open(nltk_vectorizer_file, "rb")) # Load command
    train_data_features = vectorizer.transform(uploaded_df['Content'])
    x_uploaded=pd.DataFrame(train_data_features.toarray())
    predictions=nltk_model.predict(x_uploaded)
    uploaded_df['Sentiment'] = predictions


def lstm_user_data_test(lstm_model_file, encoder_file, uploaded_df, data_folder, name_list):
    lstm_model = load_model(lstm_model_file)
    encoder = pickle.load(open(encoder_file, "rb"))
    sound_df=pd.DataFrame(columns=['fileName', 'soundFile'])
    sound_df=import_sound_files(uploaded_df, sound_df, data_folder)
    uploaded_df=pd.concat([uploaded_df, sound_df], axis=1, join='outer')
    sound_df.drop(sound_df.index, inplace=True)

    uploaded_df['Sentiment'] = uploaded_df['Sentiment'].astype(float)
    uploaded_df['soundFile'] = np.array(uploaded_df['soundFile'])
    for i in range(len(uploaded_df['soundFile'])):
        uploaded_df['soundFile'][i] = np.append(uploaded_df['soundFile'][i], uploaded_df['Sentiment'][i])
    x_uploaded=uploaded_df['soundFile']
    x_uploaded = [i for i in x_uploaded]
    x_uploaded = np.array(x_uploaded)
    x_uploaded = np.expand_dims(x_uploaded, -1)
    predictions=lstm_model.predict(x_uploaded)

    if len(predictions)!=1:
        saved_path=prediction_table(predictions, name_list, encoder)
        return saved_path
    else:
        saved_path=prediction_percentages(encoder, predictions, name_list)
        return saved_path


def prediction_table(predictions, name_list, encoder):
    fig, ax = plt.subplots()
    ax.set_title("Uploaded Files' Analysis")
    original_predictions = encoder.inverse_transform(predictions)
    table_data = [["File Names", "Emotions"]]
    for i in range (len(predictions)):
        table_data.append([name_list[i], original_predictions[i]])

    table = ax.table(cellText = table_data, loc = 'center', cellLoc='left')
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
          cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    saved_path = 'graphs/table.png'
    plt.savefig(saved_path) 
    #plt.show()
    return saved_path


def prediction_percentages(encoder, predictions, name_list):
    possible_encodings = encoder.categories_[0] # First element is a list of possible emotions
    softmax_probabilities = tf.nn.softmax(predictions) 

    for i, prediction in enumerate(softmax_probabilities): # Loops through every element and also keeps the index
        decoded_emotion = possible_encodings[np.argmax(prediction)]
        print(f"\nPrediction of file: {i}: {decoded_emotion}")
        
        # Creating a figure and axis
        fig, ax = plt.subplots()

        # Creating the bar chart
        bars = ax.bar(possible_encodings, prediction, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title(f'Prediction {i+1}: {decoded_emotion}, file name: {name_list}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Adding percentage labels on top of each bar
        for bar, percentage in zip(bars, prediction):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, '%.2f%%' % (percentage * 100),
                    ha='center', va='bottom')
            
        saved_path = 'graphs/percentages.png'
        plt.savefig(saved_path) 
        #plt.show()
        return saved_path


def training_process(nltk_model_file, lstm_model_file, nltk_vectorizer_file, encoder_file):
    train_data_folder = "dataset/training"
    test_data_folder = "dataset/test"
    val_data_folder = "dataset/val"

    """unprocessed_train_df=pd.read_excel('excel_files/train_excel.xlsx')
    unprocessed_test_df=pd.read_excel('excel_files/test_excel.xlsx')
    unprocessed_val_df=pd.read_excel('excel_files/val_excel.xlsx')

    train_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])
    test_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])
    val_df=pd.DataFrame(columns=['fileName','Emotion','Content','Sentiment'])

    # Preparing the dfs
    train_df=dataframe_preprocessing(train_data_folder, unprocessed_train_df)
    test_df=dataframe_preprocessing(test_data_folder, unprocessed_test_df)
    val_df=dataframe_preprocessing(val_data_folder, unprocessed_val_df)"""

    #Training nltk model
    """nltk_model, vectorizer=text_analysis_training(train_df, test_df)
    pickle.dump(nltk_model, open(nltk_model_file, "wb"))
    pickle.dump(vectorizer, open(nltk_vectorizer_file, "wb"))"""
    #nltk_model = pickle.load(open("nltk_model.dat", "rb")) # Load command

    # Projenin sonunda burdan
    """formatter = {'float_kind':lambda x: "%.6f," % x}

    train_df['soundFile'] = train_df['soundFile'].apply(lambda x: np.array2string(x, formatter=formatter))
    test_df['soundFile'] = test_df['soundFile'].apply(lambda x: np.array2string(x, formatter=formatter))
    val_df['soundFile'] = val_df['soundFile'].apply(lambda x: np.array2string(x, formatter=formatter))

    train_df.to_excel('saved_excels/train_df.xlsx', index=False)
    test_df.to_excel('saved_excels/test_df.xlsx', index=False)
    val_df.to_excel('saved_excels/val_df.xlsx', index=False)"""

    train_df=pd.read_excel('saved_excels/train_df.xlsx')
    test_df=pd.read_excel('saved_excels/test_df.xlsx')
    val_df=pd.read_excel('saved_excels/val_df.xlsx')
    # Buraya kadar sil

    # Training the LSTM model
    lstm_model,lstm_encoder = lstm_model_training(train_df, test_df, val_df, 55)
    lstm_model.save(lstm_model_file)
    pickle.dump(lstm_encoder, open(encoder_file, "wb"))

    #lstm_model = load_model(lstm_model_file) # Load command


def record_audio(duration, fs, filename, status_label):
    status_label.config(text="Recording process has started. Please speak...", fg="blue")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    sf.write(filename, recording, fs)
    status_label.config(text="Recording process has been completed.", fg="green")


def analyze_emotion(folder, graph_label, result_label, nltk_model_file, nltk_vectorizer_file, lstm_model_file, encoder_file):
    result_label.config(text="Extracting emotions from sound files...", fg="blue")
    uploaded_df, name_list=user_upload_files(folder)
    nltk_user_data_test(nltk_model_file, nltk_vectorizer_file, uploaded_df)
    graph_path = lstm_user_data_test(lstm_model_file, encoder_file, uploaded_df, folder, name_list)
    
    # Doğruluk oranını hesaplayıp gösterme
    graph_label.config(text="Graph:")
    graph_label.img = tk.PhotoImage(file=graph_path)
    graph_label.config(image=graph_label.img)

    result_label.config(text=f"Result Graph: ", fg="green")


def play_audio(filename):
    data, fs = librosa.load(filename, sr=None)
    sd.play(data, fs)
    sd.wait()

    
# GUI uygulaması
class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Extraction From Sound Application")
        self.root.configure(bg="#f0f0f0")

        self.load_button = tk.Button(self.root, text="Select Folder", command=self.load_audio, font=("Helvetica", 12), bg="#4CAF50", fg="white", bd=2)
        self.load_button.pack(pady=10)

        self.record_button = tk.Button(self.root, text="Record Your Sound", command=self.record_audio, font=("Helvetica", 12), bg="#008CBA", fg="white", bd=2)
        self.record_button.pack(pady=5)

        self.play_button = tk.Button(self.root, text="Listen Recorded Sound", command=self.play_recorded_audio, font=("Helvetica", 12), bg="#f44336", fg="white", bd=2, state="disabled")
        self.play_button.pack(pady=5)

        self.analyze_button = tk.Button(self.root, text="Extract Emotion", command=self.analyze_emotion, font=("Helvetica", 12), bg="#FFC107", fg="white", bd=2, state="disabled")
        self.analyze_button.pack(pady=5)

        self.status_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.status_label.pack(pady=5)

        self.duration_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.duration_label.pack(pady=5)

        self.accuracy_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.accuracy_label.pack(pady=5)

        self.graph_label = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.graph_label.pack(pady=5)

    def load_audio(self):
        filename = filedialog.askdirectory(initialdir=os.getcwd(), title="Select a Directory")
        self.audio_file = filename
        if self.audio_file:
            self.analyze_button.config(state="normal")
            self.play_button.config(state="normal")

    def record_audio(self):
        duration = 10  
        fs = 44100  
        temp_filename = "recorded_files/output_audio.wav"
        threading.Thread(target=record_audio, args=(duration, fs, temp_filename, self.status_label)).start()
        self.audio_file = "recorded_files"
        self.analyze_button.config(state="normal")
        self.play_button.config(state="normal")

    def analyze_emotion(self):
        nltk_model_file = "models/nltk_model.dat"
        lstm_model_file = "models/lstm_model.keras"
        nltk_vectorizer_file = "models/vectorizer.dat" 
        encoder_file = "models/lstm_encoder.dat"

        if hasattr(self, 'audio_file'):
            threading.Thread(target=analyze_emotion, args=(self.audio_file, self.graph_label, self.status_label, nltk_model_file, nltk_vectorizer_file, lstm_model_file, encoder_file)).start()
        else:
            self.status_label.config(text="Please upload files to use this function.", fg="red")

    def play_recorded_audio(self):
        if hasattr(self, 'audio_file'):
            if self.audio_file == 'recorded_files':
                temp = self.audio_file
                self.audio_file = "recorded_files/output_audio.wav"
                threading.Thread(target=play_audio, args=(self.audio_file,)).start()
                self.audio_file = temp
            else:
                self.status_label.config(text="This function only works for recorded .wav files", fg="red")
        else:
            self.status_label.config(text="Please record a .wav file first.", fg="red")
        

def run_training():
    nltk_model_file = "models/nltk_model.dat"
    lstm_model_file = "models/lstm_model.keras"
    nltk_vectorizer_file = "models/vectorizer.dat" 
    encoder_file = "models/lstm_encoder.dat"
    training_process(nltk_model_file, lstm_model_file, nltk_vectorizer_file, encoder_file)


if __name__ == "__main__":
    model = whisper.load_model("base.en")
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()
    #run_training()