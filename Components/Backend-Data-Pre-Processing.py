#import dataset we are going to use
import datasets 
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()
df.describe()

#remove unnecessary information
df.drop(df.iloc[:, 15:131], inplace=True, axis=1)
df = df.drop(["annotator_id"], axis=1)

#rescale annotated scores from (0,5) to (0,1)
import pandas as pd
df_need_to_rescale = df.drop(["comment_id", "hate_speech_score", "text"], axis=1)
df_norm = (df_need_to_rescale-df_need_to_rescale.min())/(df_need_to_rescale.max()-df_need_to_rescale.min())
df_norm = pd.concat((df.comment_id, df_norm, df.hate_speech_score, df.text), 1)
