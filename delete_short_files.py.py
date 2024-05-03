import pandas as pd
import numpy as np
import os

# Can be changed to train, test or val
dataset_folder= 'dataset/val'
document_df=pd.read_excel('excel_files/val_excel.xlsx')

unprocessed_length = len(document_df)
file_names=[]

for i in range(len(document_df)):
    line = str(document_df['Utterance'][i])
    #line = document_df['Utterance'][i]

    splitted_line = line.split()
    
    if len(splitted_line)<=1:
        file_name = f"{dataset_folder}/{document_df['fileName'][i]}.wav"
        document_df.drop(i, inplace=True)


        if os.path.exists(file_name):
            os.remove(file_name)
            document_df.drop(i, inplace=True)
            print("Deleted")
        else:
            print("Path does not exists!")

document_df = document_df.reset_index(drop=True)
document_df.to_excel("Final_Result.xlsx")
print("Length before preprocessing = ",unprocessed_length)
print("Length after preprocessing = ", len(document_df))
print("Difference = ", (unprocessed_length-len(document_df)))

        






