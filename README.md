# Emotion Extraction From Sound and Meaning
## 1) Description
This project, extracts emotion from sound by looking at the meaning of sentence and sound features. Model can classify the emotions of fear, anger, disgust, joy, surprise, sadness and neutral. 
<br>As dataset, MELD (Multimodal EmotionLines Dataset) is used. We especially choose this dataset, because we need different emotions with meaningful sentences for sentiment analysis to work properly.
### 1.1) Technologies Used
WhisperAI is used to transcirbe text from sound for sentiment analysis. 
<br>NLTK model is used to do sentiment analysis from transcribed text, which gives possible results of positive, negative and neutral.
<br>LSTM model is used to do emotion analysis. Model uses sentiment and sound file's features to determine which emotion the sound file belongs to.

## 2) How to Install and Run The Project
Firstly, dataset must be downloaded for the project to run correctly. Since dataset is too big to upload to GitHub, it is uploaded to Google Drive. 
<br>Here's the link: https://drive.google.com/file/d/1547d1dz2_kgBUKx-AHKC110-18PeNbVl/view?usp=drive_link
<br>After downloading the zip file in the drive link, dataset folder must be extracted to project folder.
<br>Secondly, WhisperAI does not support Python versions that are newer than 3.9.9. So, we recommend users to use version 3.9.9. Otherwise, project can't transcribe the text from sound.

## 3) How to Use The Project
Project has 3 different Python files.
### 3.1) Deleting Short Sound Files
"delete_short_files.py" is a Python file that deletes sound files which uses only one word to form a sentence. This file clears dataset from sound file's with less meaning.
### 3.2) Model Optimization
"model_optimization" uses TensorBoard to find the optimal hyperparameters for LSTM model. Results are saved as log files, so they can be examined if wanted.
### 3.3) Main Program
"main_file.py" used to run the main project. It has a simple design of the application. User can upload files and record its sound for the emotion extraction process. Resulting graphs can be seen in Graph folder in the project. Also, user can see the performance of the model by running training function.

## 4) Results


