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
"main_file.py" used to run the main project. It has a simple application UI. User can upload files and record its sound for the emotion extraction process. After the extraction process, resulting graphs will be seen on the application screen. Resulting graphs is saved in the Graph folder of the project. Also, user can see the performance of the model by running training function.

## 4) Results
### 4.1) Model Optimization
Accuracy results of all of the models in optimization process:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/optimization_graph1.png>
<br>From all of the models, two of the most accurate models are choosen. Accuracy of these two models:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/optimization_graph2.png>
<br>Loss of these two models:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/optimization_graph3.png>

### 4.2) Training and Test Set Results
Accuracy results of the trained model:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/accuracy.png>
Loss of the trained model:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/loss.PNG>
Confusion matrix of the test set:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/confusion_matrix.png>
Scores of the model:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/scores.png>

### 4.3) Uploaded Files and Their Real Results' Comparison
User uploads more than one files:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/table.png>
Real values of these files are:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/real_values%20of_table.PNG>
As we can see from the comparison of these results, anger and sadness, surprise and joy can be mixed up. We can come up with this result by looking at this comparison and confusion matrix

### 4.4) Recorded Files' Results
If there is only one sound file to extract emotion, graph changes. This graph shows the every emotion's percentage of the sound file. Here are some of the results:
<br>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/percentages.png>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/percentages_anger.png>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/percentages_sadness.png>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/percentages_joy.png>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/percentages_anger1.png>

### 4.4) GUI
Here are some images showing the GUI:
<br>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/UI_image.PNG>
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/UI_image_table.PNG>

## 4.5) Comparing Accuracy with Other Projects
Accuracy of the other projects that are done with the same dataset:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/accuracy_of_other_projects.png>
Accuracy of our project:
<img src=https://github.com/MBToker/Emotion_extraction_from_sound_and_meaning/blob/main/graphs/accuracy.png>
As we can see from the comparison above, other projects' accuracy results are below %70. This project's accuracy changes between %78-%80. By adding the sentiment analysis to the emotion extraction we build a better model.













