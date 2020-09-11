'''
prepare data for style token extraction from gst model
we need data set of type input_text_with_punctuation -> style_token pair
python -m spacy download en_core_web_sm
'''

import os,sys
import json
import re
import spacy
from difflib import SequenceMatcher
import logging
from termcolor import colored, cprint

nlp = spacy.load('en_core_web_sm')

class caption(object):

    def __init__(self,text,start,end):
        self.text = text
        self.start = start
        self.end = end

    def __init__(self,d):
        self.text = d['text']
        self.start = d['start']
        self.end = d['end']

def clean_text(text):

    # remove single newline
    # text = re.sub('(?<=[^\s])\n(?=[^\s])',' ',text)
    text = re.sub('\n',' ',text)
    text =re.sub('\s+',' ',text)
    return text

def align_text(text_path, caption_path, new_caption_path=None,debug=False):

    with open(text_path,'r') as f:
        text = f.read()

    with open(caption_path) as f:
        captions = json.load(f)

    captions = [caption(w) for cap in captions for w in cap['words']]
    caption_words = [cap.text.lower() for cap in captions]

    # debug
    # text = text[:5000]
    # text = clean_text(text)
    doc = nlp(text[:100000])
    # sents = list(doc.sents)
    text_tokens = [w for w in doc] # spacy's tokens
    text_words = [(i,w.text.lower()) for i,w in enumerate(text_tokens) if w.is_alpha] # text version of tokens
    index_textword2texttoken = {j:text_words[j][0] for j in range(len(text_words))}
    index_texttoken2textword = {v:k for k,v in index_textword2texttoken.items()}
    text_words = [pair[1] for pair in text_words]

    # align
    matched_block_ids = [] # block is word
    # aligned_sents = [] #[(span_i,start_i,end_i),...]
    s = SequenceMatcher(None, caption_words, text_words)  # check
    for block in s.get_matching_blocks():
        matchIndex1 = block[0]
        matchIndex2 = block[1]
        count = block[2]

        # start = captions[matchIndex1].start
        # end = captions[matchIndex1+count].end
        for i in range(block[2]):
            matched_block_ids.append((matchIndex1 + i, matchIndex2 + i))
    match_wordindex_cap2text = dict(matched_block_ids) # matching index of cap word in text_word list
    match_wordindex_text2cap = {v:k for k,v in match_wordindex_cap2text.items()}
    aligned_captions = []
    for sent in doc.sents:
        start = None
        end = None
        # debug alignment
        start_cap_id = None
        end_cap_id = None
        for token in sent:
            if not token.is_alpha: continue
            textword_index = index_texttoken2textword[token.i]
            # matched cap index
            capword_index = match_wordindex_text2cap.get(textword_index,None)
            if capword_index is not None:
                if start is None: start = captions[capword_index].start
                end = captions[capword_index].end

                if start_cap_id is None: start_cap_id = capword_index
                end_cap_id = capword_index

        sent_text = clean_text(sent.text)
        if start is None:
            aligned_captions.append({"text":sent_text,'start':'00:00:00.000','end':'00:00:00.000','match':False})
            if debug: cprint('%s  --> ...'%sent_text,'red')
        else:
            aligned_captions.append({"text":sent_text,'start':start,'end':end,'match':True})
            if debug: print('%s  -->  %s'%(colored(sent_text,'green'),' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])))

        if not new_caption_path:
            new_caption_path = os.path.splitext(caption_path)[0]+'_book.json'
        with open(new_caption_path,'w') as f:
            json.dump(aligned_captions,f,indent=4)

    return new_caption_path


if __name__ == '__main__':

    if len(sys.argv) == 1:
        text_path='/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai.txt'
        caption_path = '/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai.en.json'
    else:
        text_path = sys.argv[1]
        caption_path = sys.argv[2]
    # text_path = '/home/rajesh/work/limbo/data/yt/audiobooks/test.txt'
    # caption_path = '/home/rajesh/work/limbo/data/yt/audiobooks/test.json'
    align_text(text_path,caption_path,None,True)
