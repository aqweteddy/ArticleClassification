import time
import requests
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import urllib

NUM_ARTICLE = 100
df = pd.read_csv('../fetch_data/merge_train.csv')
df = shuffle(df)
texts = (df['title'] + ' ' + df['text']).to_list()
answers = df['category'].to_list()

start = time.time()

correct = 0
for text, ans in tqdm(zip(texts[:NUM_ARTICLE], answers[:NUM_ARTICLE]), desc='Requesting'):
    arg = urllib.parse.urlencode({'text': text, 'topk': 2})
    result = requests.get(f'http://127.0.0.1:5000/infer_one_class?{arg}').json()
    correct += (1 if ans in result['class'] else 0)


print(f'cost {time.time() - start} secs')
print(f'{correct}/{NUM_ARTICLE}, accuracy: {correct/NUM_ARTICLE}')