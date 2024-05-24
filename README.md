# Emotion Extraction From Sound and Meaning
## Description
This project, extracts emotion from sound by looking at the meaning of sentence and sound features. Model can classify the emotions of fear, anger, disgust, joy, surprise, sadness and neutral. 
#### Technologies Used
WhisperAI was used to transcirbe text from sound for sentiment analysis. 
<br>NLTK model was used to do sentiment analysis from transcribed text, which gives possible results of positive, negative and neutral.
<br>LSTM model was used to do emotion analysis. Model uses sentiment and sound file's features to determine which emotion the sound file belongs to.

## How to Install and Run The Project
Firstly, dataset must be downloaded for the project to run correctly. Since dataset is too big to upload to GitHub, it is uploaded to Google Drive. 
<br>Here's the link: https://drive.google.com/file/d/1547d1dz2_kgBUKx-AHKC110-18PeNbVl/view?usp=drive_link
<br>After downloading the zip file in the drive link, dataset folder must be extracted to project folder.
<br>Secondly, WhisperAI does not support Python versions that are newer than 3.9.9. So, we recommend users to use version 3.9.9. Otherwise, project can't transcribe the text from sound.
