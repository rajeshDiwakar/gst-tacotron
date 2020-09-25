'''
fix # and ** at the end of values
'''

import json
import glob
import os,sys
from http import client
from collections import Counter
import requests
import time
import random

ftrans_cache = 'transliteration.cache'
if os.path.isfile(ftrans_cache):
    with open(ftrans_cache,'r',encoding='utf-8') as f:
        trans_cache = json.load(f)
        trans_cache = dict([(k,v.split('|')) for k,v in trans_cache.items() ])
else:
        trans_cache = {}

delays = [i/10 for i in range(6)]
def get_transliteration(word):
#...     conn = client.HTTPSConnection('inputtools.google.com')
#...     conn.request('GET', '/request?text=' + word + '&itc=fr-t-i0-und&num=5&cp=0&cs=1&ie=utf-8&oe=utf-8&app=test')
#...     res = conn.getresponse()
    try:
            ret = trans_cache[word.lower()]
            return ret
    except KeyError:
        res = requests.get('http://www.google.com/inputtools/request?text=%s&ime=transliteration_en_hi&num=5&cp=0&cs=0&ie=utf-8&oe=utf-8&app=jsapi&uv'%word)

        if res.status_code==200:
            time.sleep(random.sample(delays,1)[0])
            ret = res.json()[1][0][1]
            trans_cache[word.lower()] = ret
            return ret

        else:
            print('>>  ERROR ')
            print(res.text)
            return []
    except Exception as e:
        print(e)
        return []


def split_words(path):
    with open(path,encoding='utf-8') as f:
        data = json.load(f)
    tokens = []
    for d in data:
        text = d['text']
        tokens.extend(text.split(' '))
    return tokens

search_dir = sys.argv[1]
files = glob.glob(os.path.join(search_dir,'*.json'))
tokens = []
for file in files:
    tokens.extend(split_words(file))

tf = Counter(tokens)
tf = list(tf.items())
tf.sort(key=lambda x:x[1],reverse=True)
with open(os.path.join(search_dir,'tokens.tsv'),'w',encoding='utf-8') as f:
    f.write('\n'.join(['%d\t%s\t%s'%(x[1],x[0],get_transliteration(x[0])) for x in tf]))

with open(ftrans_cache,'w',encoding='utf-8') as f:
    trans_cache = dict([(k,'|'.join(v)) for k,v in trans_cache.items() ])
    json.dump(trans_cache,f)
