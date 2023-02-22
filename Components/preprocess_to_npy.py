import datasets 
import numpy as np
import numpy as np
from langdetect import detect_langs
from pandarallel import pandarallel 
import pandas as pd


#import dataset we are going to use
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()

#use langdetect to detect language with parallelization; only keep english comments
pandarallel.initialize()
df['lang'] = df.parallel_apply(lambda x: detect_langs(x['text']), axis=1)
df['lang'] = df['lang'].apply(lambda x: x[0].lang)
df = df[df['lang'] == 'en']
df = df.drop(['lang'], axis=1)

#remove unnecessary information
df.drop(df.iloc[:, 15:131], inplace=True, axis=1)
df = df.drop(["annotator_id"], axis=1)

#rescale annotated scores from (0,5) to (0,1)
df_need_to_rescale = df.drop(["comment_id", "hate_speech_score", "text"], axis=1)
df_norm = (df_need_to_rescale-df_need_to_rescale.min())/(df_need_to_rescale.max()-df_need_to_rescale.min())
df_norm = pd.concat((df.comment_id, df_norm, df.hate_speech_score, df.text), 1)

# remove non ascii characters from text 
df_norm['text'] = df_norm['text'].apply(lambda x: ''.join([i if ord(i) < 128 else '' for i in x]))

#convert to numpy array
tmp_np_arr = df_norm.to_numpy()

#save numpy array to file
np.save('hate_speech.npy', tmp_np_arr)