"""
Подробное описание данного скрипта в тетрадке data_union.py
"""

import os

import pandas as pd

speech_path = './/speech/'
captioning_path = './/captioning/'

output_speech_path = './/speech_1.csv'
output_captioning_path = './/union_1.csv'

concated_speech_frame = None
concated_captionin_frame = None

for dirname, _, filenames in os.walk(speech_path):
    i = 0
    for filename in filenames:
        if '.csv' in filename:
            if i == 0:
                df_path = str(os.path.join(dirname, filename))
                concated_speech_frame = pd.read_csv(df_path)
                i += 1
            else:
                df_path = str(os.path.join(dirname, filename))
                temp_df = pd.read_csv(df_path)
                concated_speech_frame = pd.concat([concated_speech_frame, temp_df])

for dirname, _, filenames in os.walk(captioning_path):
    i = 0
    for filename in filenames:
        if '.csv' in filename and filename != 'test_0.csv':
            if i == 0:
                df_path = str(os.path.join(dirname, filename))
                concated_captionin_frame = pd.read_csv(df_path)
                i += 1
            else:
                df_path = str(os.path.join(dirname, filename))
                temp_df = pd.read_csv(df_path)
                concated_captionin_frame = pd.concat([concated_captionin_frame, temp_df])

text_link_map = {

}

for index, row in concated_speech_frame.iterrows():
    row_link = row['link']
    row_text = row['text']
    text_link_map[row_link] = row_text

def find_speech_by_link(row):
    return text_link_map.get(row['link'])


#concated_captionin_frame['text'] = concated_captionin_frame.apply(find_speech_by_link, axis=1)

# concated_captionin_frame.to_csv(output_captioning_path, index=False)
# concated_speech_frame.to_csv(output_speech_path, index=False)
