'''
prepare data for style token extraction from gst model
we need data set of type input_text_with_punctuation -> style_token pair
python -m spacy download en_core_web_sm

youtube captions only use : |-[]0-9a-zA-Z '|

'''

import os,sys,glob
import json
import re
import spacy
from difflib import SequenceMatcher
import logging
from termcolor import colored, cprint
from vtt2json import vtt2json

nlp = spacy.load('en_core_web_sm')
# nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
emphasis = {'á':'a','é':'e','í':'i','ñ':'n','ó':'o','ú':'u','â':'a','&':'and'}# present in lyrics from web
USE_FORCED_SENT_ALIGNMENT=False # for audiobooks
USE_STRICT_MATCH=True # FOR SONGS. SELECT ONLY MATCHING PARTS
MAX_WORD_ERROR = 1


def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "\n":
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe(set_custom_boundaries, before="parser")

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

def clean_lyrics(text):
    text = re.sub('\[.*\]','',text)
    'á é í ñ ó ú'
    for c in emphasis:
        text  = re.sub(c,emphasis[c],text)
    text = re.sub("[^a-zA-Z0-9-' \n]",'',text)
    text = re.sub(' +',' ',text)
    text = re.sub('\n+','\n',text)

    return text

def is_valid_word(text):
    return re.match("[0-9a-zA-Z-']",text)

def iou(text1,text2):
    text1 = set(text1.lower().split(' '))
    text2 = set(text2.lower().split(' '))
    if len(text1)+len(text2)==0:return 0
    return len(text1.intersection(text2))/float(len(text1.union(text2)))

def span2sent(span):
    ret = []
    sent = []
    w=span[0]
    # while w:
    for w in span:
        if w.is_sent_start:
            if len(sent):
                ret.append(sent)
                sent= []
        sent.append(w)
    if len(sent): ret.append(sent)

    return ret

def align_text(text_path, caption_path, new_caption_path=None,debug=False):

    with open(text_path,'r') as f:
        text = f.read()
        # text = '\n'.join([t.strip() for t in text.split('\n') if t.strip()])

    with open(caption_path) as f:
        captions = json.load(f)

    captions = [caption(w) for cap in captions for w in cap['words']]
    caption_words = [cap.text.lower() for cap in captions]

    # debug
    # text = text[:5000]
    # text = clean_text(text)
    # remove emphasis chars
    text = clean_lyrics(text)
    lyricslen = len(text)
    doc = nlp(text[:100000])
    position = [token.i for token in doc if token.i!=0 and "'" in token.text]
    with doc.retokenize() as retokenizer:
        for pos in position:
            try:
                retokenizer.merge(doc[pos-1:pos+1])
            except ValueError:
                print('error while merging tokens:',doc[pos-1:pos+1])

    # sents = list(doc.sents)
    text_tokens = [w for w in doc] # spacy's tokens
    # text_words = [(i,w.text.lower()) for i,w in enumerate(text_tokens) if w.is_alpha] # text version of tokens
    text_words = [(i,w.text.lower()) for i,w in enumerate(text_tokens) if is_valid_word(str(w))] # text version of tokens
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
    if debug:
        print(caption_words)
        print(text_words)
        # col1 = []
        # col2 = []
        # arrow = []
        # prev_i = 0
        # prev_j = 0
        # for i,j in matched_block_ids:
        #     col1.extend(caption_words[prev_i:i])
        #     col2.extend(text_words[prev_j:j])
        #     prev_i = i
        #     prev_j = j
        #     lendiff = len(col1)-len(col2)
        #     if lendiff > 0:
        #         col2.extend(['---' for _ in range(lendiff)])
        #     elif lendiff < 0:
        #         lendiff = -lendiff
        #         col1.extend(['---' for _ in range(lendiff)])
        #     arrow.append()
        # print_alignment(caption_words)
        for i,j in matched_block_ids:
            print('%d:%s --> %d:%s'%(i,caption_words[i],j,text_words[j]))
    aligned_captions = []
    if USE_FORCED_SENT_ALIGNMENT:
        for sent in doc.sents:
            start = None
            end = None
            # debug alignment
            start_cap_id = None
            end_cap_id = None
            start_text_i = None # spacy's index for text
            end_text_i = None
            for token in sent:
                # if not token.is_alpha: continue # only for song
                if not is_valid_word(str(token)): continue
                textword_index = index_texttoken2textword[token.i]
                # end_text_i = index_texttoken2textword[token.i]

                # matched cap index
                capword_index = match_wordindex_text2cap.get(textword_index,None)
                if capword_index is not None:
                    end_text_i = token.i
                    if start_text_i is None:
                        start_text_i = token.i

                    if start is None: start = captions[capword_index].start
                    end = captions[capword_index].end

                    if start_cap_id is None: start_cap_id = capword_index
                    end_cap_id = capword_index

            sent_text = clean_text(sent.text)
            text = str(doc[start_text_i:end_text_i+1] ) if start_text_i is not None else ''
            if start is None:
                aligned_captions.append({"text":sent_text,'start':'00:00:00.000','end':'00:00:00.000','match':False,'match_text':''})
                if debug: cprint('%s  --> ...'%sent_text,'red')
            else:
                match_text = ' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])
                # aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'sent_text':sent_text})
                dist = iou(text,match_text)
                if end_cap_id-start_cap_id+1<3:
                    print(colored('Discarding due to len < 3\n%s'%match_text,'red',on_color='on_blue'))
                elif dist < 0.7:
                    print(colored('Discarding due to iou(=%.2f)<0.7\n%s U %s'%(dist,text, match_text),'red',on_color='on_blue'))
                else:
                    aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'match_text':match_text})
                if debug: print('%s  -->  %s'%(colored(sent_text,'green'),' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])))

            if not new_caption_path:
                new_caption_path = os.path.splitext(caption_path)[0]+'_book.json'
            with open(new_caption_path,'w') as f:
                json.dump(aligned_captions,f,indent=4)

        return new_caption_path
    elif 0 and USE_STRICT_MATCH:
        # for i,(capword_index,textword_index) in enumerate(matched_block_ids):

        for sent in doc.sents:
            start = None
            end = None
            # debug alignment
            start_cap_id = None
            end_cap_id = None
            start_text_i = None # spacy's index for text
            end_text_i = None
            word_error = 0
            for token in sent:
                # if not token.is_alpha: continue # only for song
                if not is_valid_word(str(token)): continue
                textword_index = index_texttoken2textword[token.i]
                # end_text_i = index_texttoken2textword[token.i]

                # matched cap index
                capword_index = match_wordindex_text2cap.get(textword_index,None)
                if capword_index is not None:
                    end_text_i = token.i
                    if start_text_i is None:
                        start_text_i = token.i

                    if start is None:
                        start = captions[capword_index].start
                        word_error = 0
                    end = captions[capword_index].end
                    if start_cap_id is None: start_cap_id = capword_index
                    end_cap_id = capword_index
                else:
                    if start is not None:
                        word_error += 1

                if word_error > MAX_WORD_ERROR:

                    sent_text = clean_text(sent.text)
                    text = str(doc[start_text_i:end_text_i+1] ) if start_text_i is not None else ''
                    if start is None:
                        # aligned_captions.append({"text":sent_text,'start':'00:00:00.000','end':'00:00:00.000','match':False,'match_text':''})
                        if debug: cprint('%s  --> ...'%sent_text,'red')
                    else:
                        match_text = ' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])
                        # aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'sent_text':sent_text})
                        dist = iou(text,match_text)
                        if end_cap_id-start_cap_id+1<3:
                            print(colored('Discarding due to len < 3\n%s'%match_text,'red',on_color='on_blue'))
                        # elif dist < 0.7:
                        #     print(colored('Discarding due to iou(=%.2f)<0.7\n%s U %s'%(dist,text, match_text),'red',on_color='on_blue'))
                        else:
                            aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'match_text':match_text})
                        if debug: print('%s  -->  %s'%(colored(text,'green'),' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])))

                    start = None
                    end = None
                    # debug alignment
                    start_cap_id = None
                    end_cap_id = None
                    start_text_i = None # spacy's index for text
                    end_text_i = None
                    word_error = 0

            sent_text = clean_text(sent.text)
            text = str(doc[start_text_i:end_text_i+1] ) if start_text_i is not None else ''
            if start is None:
                # aligned_captions.append({"text":sent_text,'start':'00:00:00.000','end':'00:00:00.000','match':False,'match_text':''})
                if debug: cprint('%s  --> ...'%sent_text,'red')
            else:
                match_text = ' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])
                # aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'sent_text':sent_text})
                dist = iou(text,match_text)
                if end_cap_id-start_cap_id+1<3:
                    print(colored('Discarding due to len < 3\n%s'%match_text,'red',on_color='on_blue'))
                elif dist < 0.7:
                    print(colored('Discarding due to iou(=%.2f)<0.7\n%s U %s'%(dist,text, match_text),'red',on_color='on_blue'))
                else:
                    aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'match_text':match_text})
                if debug: print('%s  -->  %s'%(colored(text,'green'),' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])))

            if not new_caption_path:
                new_caption_path = os.path.splitext(caption_path)[0]+'_book.json'
            with open(new_caption_path,'w') as f:
                json.dump(aligned_captions,f,indent=4)

        return new_caption_path
    elif USE_STRICT_MATCH:
        # first group on continuous cap ids
        continuous_caps = []
        curr_blocks = []
        for k,(i,j) in enumerate(matched_block_ids):
            if k == len(matched_block_ids)-1:
                curr_blocks.append((i,j))
            else:
                if matched_block_ids[k+1][0]-i -1> MAX_WORD_ERROR:
                    curr_blocks.append((i,j))
                    continuous_caps.append(curr_blocks)
                    curr_blocks = []
                else:
                    curr_blocks.append((i,j))

        if len(curr_blocks):
            continuous_caps.append(curr_blocks)

        # now group on text ids
        continuous_texts = []

        for block in continuous_caps:
            curr_blocks = []
            for k,(i,j) in enumerate(block):
                if k==len(block)-1:
                    curr_blocks.append((i,j))
                else:
                    if block[k+1][1]-j-1>MAX_WORD_ERROR:
                        curr_blocks.append((i,j))
                        continuous_texts.append(curr_blocks)
                        curr_blocks = []
                    else:
                        curr_blocks.append((i,j))
            if len(curr_blocks):
                continuous_texts.append(curr_blocks)



        aligned_captions = []
        for block in continuous_texts:

            if len(block)<3: continue
            start_text_i = index_textword2texttoken[block[0][1]]
            end_text_i  = index_textword2texttoken[block[-1][1]]
            text = doc[start_text_i:end_text_i+1]
            if debug:
                print('\n\n\t\t\t\t-------------\n',text)
            # for sent in text.as_doc().sents:
            for sent in span2sent(text):
            # sent = text.sent
            # if 1:
                # for tk in sent:
                #     print(tk.i,tk)
                start=end=start_text_i=end_text_i=start_cap_id=end_cap_id=None
                if debug:
                    print('\n>>> ',sent[:-1],len(sent))
                for i in range(len(sent)):
                    textword_index = index_texttoken2textword.get(sent[i].i,None)
                    capword_index = match_wordindex_text2cap.get(textword_index,None)
                    if debug: print('start: ',i,sent[i],sent[i].i,textword_index,capword_index)
                    # if textword_index:
                    #     print(text_words[textword_index])
                    if textword_index is None or capword_index is None:
                        continue
                    start = captions[capword_index].start
                    start_cap_id = capword_index
                    start_text_i = sent[i].i # spacy's index for text
                    break

                for i in range(len(sent)-1,-1,-1):
                    textword_index = index_texttoken2textword.get(sent[i].i,None)
                    capword_index = match_wordindex_text2cap.get(textword_index,None)
                    # print('end: ',i,sent[i],sent[i].i,textword_index,capword_index)
                    # if textword_index:
                    #     print(text_words[textword_index])
                    if textword_index is None or capword_index is None:
                        continue
                # textword_index = index_texttoken2textword[sent[-1].i]
                # capword_index = match_wordindex_text2cap[textword_index]
                    end = captions[capword_index].end
                    end_cap_id = capword_index
                    end_text_i = sent[i].i
                    break

                # print('ids: ',start_text_i,end_text_i,start_cap_id,end_cap_id)
                if start_text_i is None or end_text_i is None or start_cap_id is None or end_cap_id is None:
                    print(colored('ALIGNING FAILD','blue','on_red'))
                    continue
                match_text = ' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])
                text = str(doc[start_text_i:end_text_i+1] )
                if end_cap_id-start_cap_id+1<3:
                    if debug: print(colored('Discarding due to len < 3\n%s'%match_text,'red',on_color='on_blue'))
                else:
                    aligned_captions.append({"text":text,'start':start,'end':end,'match':True,'match_text':match_text})
                if debug: print('%s  -->  %s'%(colored(text,'green'),' '.join([w.text for w in captions[start_cap_id:end_cap_id+1]])))

        # print('\nPrinting captions')
        # for block in continuous_caps:
        #     print(' '.join([w.text for w in captions[block[0][0]:block[-1][0]+1] ]))
        #
        # print('\nPrinting continuous text')
        # for block in continuous_texts:
        #     print(' '.join(text_words[block[0][1]:block[-1][1]+1]))
        print('Match Percentage: %.2f'%(100*len(' '.join([cap['text'] for cap in aligned_captions]))/lyricslen) )

        if not new_caption_path:
            new_caption_path = os.path.splitext(caption_path)[0]+'_aligned.json'
        with open(new_caption_path,'w') as f:
            json.dump(aligned_captions,f,indent=4)

    else:
        raise ValueError('no alignment method specified')


if __name__ == '__main__':

    if len(sys.argv) == 1:
        text_path='/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai.txt'
        caption_path = '/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai.en.json'
    else:
        if sys.argv[1].endswith('.json') or sys.argv[1].endswith('.vtt'):
            caption_path = sys.argv[1]
            text_path = sys.argv[2]
        else:
            text_path = sys.argv[1]
            caption_path = sys.argv[2]
    if caption_path.endswith('.vtt'):
        caption_path = vtt2json(caption_path,'temp.json')
    # text_path = '/home/rajesh/work/limbo/data/yt/audiobooks/test.txt'
    # caption_path = '/home/rajesh/work/limbo/data/yt/audiobooks/test.json'
    align_text(text_path,caption_path,None,False)
