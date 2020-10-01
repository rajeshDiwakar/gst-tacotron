'''
python prepare_tts_data.py --booklist hindi/booklist_test  --start 0 --trans_cache transliteration.cache.min3.man --data_dir tts --augment rand_step

sudo apt-get install libsox-fmt-mp3

use this tool to prepare tts dataset
first collect youtube video list
download captions
do tfidf analysis on captions - find most frequent words and generated transliteration file
run this script to generate audio segments and metadata

TODO
normalise text - remove numbers
include random length texts # important
expand numbers
time
split hyphan
remove lines containing other chars like []
fix percent
handle acronyms - bjp,ndtv,nrc
'''

import os,sys,time
import logging
import argparse
import csv
import json
import youtube_dl
import re
import glob
import subprocess
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from shutil import copyfile, copy2
import librosa
import argparse
import threading
import time
import random
from random import randint

from vtt2json import livevtt2json
from book_caption import align_text
from number_utils import normalize_numbers

BLOCKED_CHANNELS = ['UCOIu6fZOViBEgvhSYqka4GQ','UCZHso3FEeHMv9bNRMbNF9_w','UCkQpNod01EA98RXnbSfifyw']

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort')
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

def expand_numbers(text):
  return normalize_numbers(text)


transliteration_cache=None

def load_transliteration_cache(path):
    global transliteration_cache
    with open(path,'r',encoding='utf-8') as f:
        transliteration_cache = json.load(f)
        transliteration_cache = dict([(k,v.split('|')) for k,v in transliteration_cache.items() ])

def load_csv(file_path):
    with open(file_path) as csvfile:
        sniffer = csv.Sniffer()
        # sniffer.delimiter = '\t'
        # dialect = sniffer.sniff(csvfile.read(10240))
        # csvfile.seek(0)
        # reader = csv.reader(csvfile, dialect)
        reader = csv.reader(csvfile, delimiter='\t')
        return [row for row in reader if len(row)] #(i,name,author,pdf_urls)

def load_audio_meta(file_path):
    meta = defaultdict(list)
    with open(file_path,encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            key = line['query']
            videos = line['videos']
            # videos.sort(key=lambda x:x['duration'],reverse=True) # sort by duration instead of popularity
            # sys.stdout.write('%50s\t\t\t%d\t'%(key[:50],len(videos)) )
            videos = [v for v in videos if v['channel_id'] not in BLOCKED_CHANNELS and v['automatic_captions'] ]
            if not (len(videos)):
                # print(0,0,0)
                continue
            durations = [v['duration'] for v in videos]
            durations.sort(reverse=True)
            min_duration = durations[0]*0.90
            videos = [v for v in videos if v['duration']>=min_duration]
            # print(len(videos),min_duration,','.join([str(i) for i in durations[:5]]))
            if key.endswith(' audiobook'):
                key = key[:-len(' audiobook')]
            elif key.endswith(' audiobook female voice'):
                key = key[:-len(' audiobook female voice')]
            # meta[key].extend(videos)
            meta[key] = videos

    return dict(meta)

def pdf2text(src,target=None):
    if not target:
        target = os.path.splitext(src)[0] + '.txt'

    cmd = 'pdftotext "%s" "%s"'%(src,target)
    ret = os.system(cmd) # 0 for success

    return target, ret

def get_bookdir(root,name):
    name = re.sub('[^a-zA-Z.-0-9]','_',name)
    return os.path.join(root,name)

def download_pdf_make_text(urls,target):
    for url in urls:
        if os.path.isfile(target):
            ret = 0
        else:
            cmd = "wget --timeout=120 -O '%s' '%s'"%(target,url)
            ret = os.system(cmd)
        if not ret:
            text_path,ret = pdf2text(target)
            if not ret:
                with open(text_path,encoding='utf-8') as f:
                    count = len(f.read().strip()) # it should have at least 100 chars
                    if count>100:
                        break

        # check if downloaded file is really pdf and log info about it

    if not os.path.isfile(target):
        raise Exception('unable to fetch: %s'%str(urls))

def download_audio_caption(vid,target_dir,args):
    # raise NotImplemented
    url = 'http://www.youtube.com/watch?v='+vid
    ydl = youtube_dl.YoutubeDL({
                            'outtmpl': os.path.join(target_dir,'%(id)s.%(ext)s'),
                            'write-auto-sub':True,
                            'sub-lang':'en',
                            'writesubtitles':False,
                            'writeautomaticsub':True,
                            'write-sub':False,
                            'quiet':True,
                            'no_warnings':True,
                            'subtitleslangs':['en'],
                            'format':'worstaudio',
                            'postprocessors': [{
                                        'key': 'FFmpegExtractAudio',
                                        'preferredcodec': 'mp3',
                                    }],
                            'cookiefile':args.cookiefile
                                })
    with ydl:
        ydl.download([url])


# def str2ms(time_str):
#     m = re.match('([0-9]+):([0-9]+):([0-9]+).([0-9]+)',time_str)
#     h,m,s,ms = m.group(1,2,3,4)
#     return (int(h)*3600 + int(m)*60 + int(s))*1000+ int(ms)


def ms2str(milliseconds):
    hh = milliseconds//3600000
    milliseconds %= 3600000
    mm = milliseconds//60000
    milliseconds %= 60000
    ss = milliseconds // 1000
    ms = int(milliseconds % 1000)
    return '%02d:%02d:%02d.%d'%(hh,mm,ss,ms)

def str2ms(time_str):
    m = re.match('([0-9]+):([0-9]+):([0-9]+).([0-9]+)',time_str)
    h,m,s,ms = m.group(1,2,3,4)
    return (int(h)*3600 + int(m)*60 + int(s))*1000+ int(ms)


'TODO'
def clean_text(text):
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = re.sub('-',' ',text)
    words = [w.strip() for w in text.split(' ') if w.strip()]
    trans = []
    for w in words:
        try:
            wtrans = transliteration_cache[w.lower()]
            if len(wtrans):
                wtrans = re.sub('[*]','',wtrans[0])
                wtrans = re.sub('#[0-9]+','',wtrans)
                if wtrans.strip():
                    trans.append(wtrans)
                else:
                    raise KeyError
            else:
                raise KeyError
        except KeyError:
            trans.append(w)
            # print('no transliteration for %s'%w)
    text = ' '.join(trans)
    return text


def split_audio(audio,caption,root,sr=22050,buffer=1800,augmentation=None,aug_factor=3,min_duration=1,max_duration=10):
    '''
    buffer: audio chunk in seconds to load to avoid multiple mem reads
    '''
    with open(caption) as f:
        captions = json.load(f)
    os.makedirs(root,exist_ok=True)
    meta = []
    # count = 0
    offset = 0 # in ms
    frames = None
    basename = os.path.basename(audio)
    basename = os.path.splitext(basename)[0]
    if basename.endswith('_mono'): basename = basename[:-5]
    if augmentation:
        if augmentation == 'rand_step':
            words = [word for cap in captions for word in cap['words']]
            aug_captions = []
            for i in range(len(words)):
                word = words[i]
                word['start'] = str2ms(word['start'])
                word['end'] = str2ms(word['end'])
            duration_steps=list(range(min_duration*1000,max_duration*1000,(max_duration-min_duration)*100)) # five steps
            exp_sample_duration = sum(duration_steps)/len(duration_steps)
            total_duration = (words[-1]['end']-words[0]['start'])
            new_duration = 0
            tstart=time.time()
            for i in range(int(aug_factor*total_duration/exp_sample_duration)):
                rand_duration=duration_steps[randint(0,len(duration_steps)-1)]
                start_i = randint(0,len(words)-5)
                end_i = None
                start = words[start_i]['start']
                end = start + rand_duration
                for i in range(start_i,len(words)):
                    if words[i]['end'] > end:
                        end_i = i
                        break
                if end_i is not None:
                    aug_captions.append({'text':' '.join([w['text'] for w in words[start_i:end_i+1]]),'start':words[start_i]['start'],'end':words[end_i]['end']})
                    new_duration += (end - start)
                # print('rand_duration: %s start_i: %s end_i: %s'%(rand_duration,start_i,end_i))
            print('original duration: %s, new_duration: %s, time-taken:%.3fs'%(ms2str(total_duration),ms2str(new_duration),time.time()-tstart))
            # print('length of augmented captions: %d, time:%.3fs'%(len(aug_captions),time.time()-tstart))
            aug_captions.sort(key=lambda x:x['start'])
        else:
            raise ValueError('unknown augmentation method %s'%augmentation)
    else:
        aug_captions = captions
    for cap in tqdm(aug_captions):
        # count += 1
        # if count >3: break
        # if not cap['match']: continue
        tstart = time.time()
        text = cap['text']
        start = cap['start']
        end = cap['end']
        if type(start) == str:
            startstr,endstr = start,end
            start = str2ms(startstr)
            end = str2ms(endstr)
        else:
            startstr = ms2str(start)
            endstr = ms2str(end)
        # reference_audio = input('Ref Audio: ').strip().strip("'")
        # ref_wav = audio.load_wav(reference_audio)
        # duration = (end-start)/1000.0
        if offset+buffer*1000 < end or offset > start or frames is None:
            offset = start
            frames,sr = librosa.core.load(audio,sr=sr,offset=start/1000.0, duration=buffer)
            # check mono
            assert len(frames.shape) == 1, 'expected monochannel audio'

        sw=2
        # assuming mono channel
        startframe=int((start-offset)*sr/1000)
        numframe = int((end-start)*sr/1000)
        clip = frames[startframe:startframe+numframe]
        # clip = split(wav,start,end)
        # wav_file.setpos(startframe)
        # clip = wav_file.readframes(numframe)
        # print('clipping from frame:%d to %d: expected %d'%(startframe,wav_file.tell(),numframe))

        clip_path = os.path.join(root,basename)
        clip_path = clip_path +'_%d-%d'%(start,end) + '.wav'
        logging.debug('Saving at: %s'%clip_path)

        # audio = struct.unpack_from("%dh" %(len(clip)//sw), clip) # audio is of librosa format
        # audio = np.array(audio)
        # audio = audio / 32768.0

        librosa.output.write_wav(clip_path, clip, sr)
        meta.append((clip_path,text,clean_text(text)))

    return meta


def convert_to_mono(audio_path,new_audio_path=None,clean=True):
    if not new_audio_path:
        new_audio_path = '_mono'.join(os.path.splitext(audio_path)) # will it work if there are no extensions

    cmd =['soxi', '-c', audio_path]
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    try:
        result = int(result.strip())
    except ValueError as e:
        raise Exception('Error while converting to mono\n%s'%result)

    if result==1:
        if clean:
            os.rename(audio_path,new_audio_path)
        else:
            copy2(audio_path,new_audio_path)
    elif result>1:
        cmd = 'sox "%s" "%s" remix %s'%(audio_path,new_audio_path,','.join([str(i) for i in range(1,result+1)]) )
        ret = os.system(cmd)
        assert ret==0

        if clean:
            os.remove(audio_path)

    return new_audio_path

class Status(object):
    def __init__(self, index=0,name='',author='',ispdf=0,istext=0,mp3='',vtt2json='',book_caption='',numchunks='',comment=''):
        self.index = index
        self.name = name
        self.author = author
        self.ispdf = ispdf
        self.istext = istext
        self.mp3 = mp3
        self.vtt2json = vtt2json
        self.book_caption = book_caption
        self.numchunks = numchunks
        self.comment = comment

    def tostr(self,separator='|'):
        return '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s'%(self.index,self.name,self.author,self.ispdf,self.istext,self.mp3,self.vtt2json,self.book_caption,self.numchunks,self.comment)

def run(args):
    load_transliteration_cache(args.trans_cache)
    booklist = args.booklist
    # audio_meta = args.audio_meta
    root = args.data_dir

    os.makedirs(root,exist_ok=True)
    # booklist = load_csv(booklist)
    with open(booklist) as f:
        booklist = [vid.strip() for vid in f.read().strip().split('\n')]
    booklist = [(i,vid,vid,'') for i,vid in enumerate(booklist)]
    end = args.end if args.end >= 0 else len(booklist)
    booklist = booklist[args.start:end]
    # audio_meta = load_audio_meta(audio_meta)
    logging.info('number of books: %s'%(len(booklist)-1))

    logfile =  os.path.splitext(args.booklist)[0]+'_%d-%d_log.csv'%(args.start,end)
    logfile = open(logfile,'w')
    logfile.write('index|name|author|ispdf|istext|mp3|vtt2json|book_caption|numchunks|comments\n')
    # print('Number of books with available audio: %d'%len(audio_meta.keys()))
    # all_keys= list(audio_meta.keys())
    for details in booklist:
        print(time.asctime(),str(details))
        mstatus = Status()
        try:
            i,name,author,pdf_urls = details[:4]
            mstatus.index, mstatus.name, mstatus.author = i,name,author
            videos = [{'id':name}] #name is video id
            # logbook = '%s|%s|%s'%(i,name,author)
            # key = '%s %s'%(name,author)
            # videos = audio_meta.get(key,[])
            # if not len(videos):
            #     # logfile.write(logfile+'-1|-1|-1')
            #     mstatus.comment += 'No video available'
            #     # logfile.write(mstatus.tostr()+'\n')
            #     print('No videos available for %s'%key)
            #     continue
            # raise Exception
            bookdir = get_bookdir(root,name)
            os.makedirs(bookdir,exist_ok=True)
            # pdf_urls = pdf_urls.split(' ')

            # bookpath = os.path.join(bookdir,'book.pdf')
            text_path = os.path.join(bookdir,'book.txt')
            # if not os.path.isfile(text_path) and not args.dry_run:
            #     download_pdf_make_text(pdf_urls, bookpath )
            # if os.path.isfile(bookpath): mstatus.ispdf = 1
            # # text_path = pdf2text(bookpath)
            # if os.path.isfile(text_path): mstatus.istext = 1
            for vid in videos[:args.max_vid_per_book]:
                audio_path = os.path.join(bookdir,vid['id']+'.mp3')
                mono_audio_path = os.path.join(bookdir,vid['id']+'_mono.mp3')
                # print(audio_path)
                if not os.path.isfile(audio_path) and not args.dry_run:
                    # download caption and audio
                    download_audio_caption(vid['id'],bookdir,args)
                if os.path.isfile(audio_path) and not args.dry_run:
                    if not os.path.isfile(mono_audio_path):
                        convert_to_mono(audio_path, clean=False)
                    mstatus.mp3 += vid['id'] +', '
                else:
                    mstatus.mp3 += 'NoAUD(%s), '%vid['id']
                    logging.error('Audio not found: %s'%audio_path)

            # fix captions
            captions = glob.glob(os.path.join(bookdir,'*.en.vtt'))
            # one metadata.csv for one book. it will store audio_path+text for all audios
            meta_csv_path = os.path.split(text_path)[0]+'/meta.csv'
            meta_filelist_path_c1 = os.path.split(text_path)[0]+'/meta_filelist_c1.txt' # for nvidia tacotron
            meta_filelist_path_c2 = os.path.split(text_path)[0]+'/meta_filelist_c2.txt' # for nvidia tacotron
            meta_csv_data = []
            isfirstline = True
            # with open(meta_csv_path,'w') as f_meta_csv:
            if 1:
                for caption in captions:
                    try:
                        vid = os.path.basename(os.path.splitext(caption)[0])
                        new_caption = os.path.splitext(caption)[0]+'.json'
                        book_caption = new_caption
                        # book_caption = os.path.splitext(caption)[0]+'_book.json'

                        if not os.path.isfile(new_caption):
                            new_caption = livevtt2json(caption,new_caption)
                        mstatus.vtt2json += '%s, '%vid

                        # if not os.path.isfile(book_caption):
                        #     # align captions with book
                        #     book_caption = align_text(text_path,new_caption,book_caption)
                        # mstatus.book_caption += '%s, '%vid

                        audio_path = new_caption[:-len('.en.json')]+'_mono.mp3'
                        # print(audio_path)
                        meta_csv = split_audio(audio_path,book_caption,root='%s/wavs'%bookdir,sr=22050,buffer=1800,augmentation=args.augment,min_duration=args.min_duration,max_duration=args.max_duration)
                        mstatus.numchunks += "%d, "%len(meta_csv)
                        meta_csv = '\n'.join(['|'.join(tpl) for tpl in meta_csv])
                        meta_csv_data.append(meta_csv)
                        # if isfirstline:
                        #     isfirstline = False
                        # else:
                        #     meta_csv = '\n'+meta_csv
                        # f_meta_csv.write(meta_csv)
                    except Exception as e:
                        mstatus.comment += str(e)
                        # logfile.write(mstatus.tostr()+'\n')
                        logging.exception(e)
        except Exception as e:
            mstatus.comment += str(e)
            # logfile.write(mstatus.tostr()+'\n')
            logging.exception(e)
        except KeyboardInterrupt:
            mstatus.comment += 'KeyboardInterrupt'
            # logfile.write(mstatus.tostr()+'\n')
        finally:
            logfile.write(mstatus.tostr()+'\n')
            try:
                with open(meta_csv_path,'w',encoding='utf-8') as f:
                    f.write('\n'.join(meta_csv_data))

                data = [line for i in meta_csv_data for line in i.split('\n')]
                data_c2 = ['|'.join(l.split('|')[0:3:2]) for l in data ]
                data_c1 = ['|'.join(l.split('|')[0:2:1]) for l in data ]
                with open(meta_filelist_path_c1,'w',encoding='utf-8') as f:
                    f.write('\n'.join(data_c1))
                with open(meta_filelist_path_c2,'w',encoding='utf-8') as f:
                    f.write('\n'.join(data_c2))
            except NameError as e:
                print(e)

    logfile.close()


if __name__ == '__main__':
    # booklist = '/home/rajesh/work/limbo/data/yt/testbooks2.tsv'
    # audio_meta = '/home/rajesh/work/limbo/data/yt/audiobook_meta.jsonl'
    # # audio_meta = '/home/rajesh/work/limbo/data/yt/audiobook_meta_test.jsonl'
    #
    # booklist = '/home/rajesh/work/limbo/data/yt/500books_pdf.tsv'
    # audio_meta = '/home/rajesh/work/limbo/data/yt/500books_audio_meta_1-65.jsonl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--booklist',required=True)
    # parser.add_argument('--audio_meta',required=True)
    parser.add_argument('--start',default=0,type=int,help='start index (including) of row to process(0 based)')
    parser.add_argument('--end',default=-1,type=int,help='last index (excluding) of row to process(counting based not the given in the list)')
    parser.add_argument('--data_dir',default='data',help='root directory path for all the data generated')
    parser.add_argument('--cookiefile',default=None,help='path to cookiefile')
    parser.add_argument('--dry_run',default=False,action='store_true',help='dry run. not downloading')
    parser.add_argument('--max_vid_per_book',default=2,help='max number of books to download per book')
    parser.add_argument('--trans_cache',required=True,help='transliteration cache')
    parser.add_argument('--augment',choices=['rand_step'],default=None,help='augmentation method')
    parser.add_argument('--min_duration',default=5,type=int)
    parser.add_argument('--max_duration',default=10,type=int)
    args=parser.parse_args()
    run(args)
