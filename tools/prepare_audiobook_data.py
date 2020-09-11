'''
python prepare_audiobook_data.py --booklist goodreads/books_pdflog.tsv --audio_meta goodreads/books_pdflog_audio_meta_0-0.jsonl --start 0 --end 10

sudo apt-get install libsox-fmt-mp3


install youtube-dl pdftotext
prepare audiobook dataset in ljspeech format
1. create a book list with index, name,author,pdf link
2. find videos on youtube with autocaption available
3. This tool will download pdf, caption and audio.
    convert pdf to text
    fix downloaded caption
    generate book caption
    generate audio clips and metadata.csv in ljspeech format
    generate filelist in mellotron format (we can use another script)

add ignore channel - x
check if file exist before download - x
convert audio to mono - add it post processor - x (using sox)
add checks like - check if downloaded pdf is correct
    aligned json is not empty

merge splitted books
normalise text - remove numbers
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
from shutil import copyfile, copy2
import librosa
import argparse

from vtt2json import livevtt2json
from book_caption import align_text

BLOCKED_CHANNELS = ['UCOIu6fZOViBEgvhSYqka4GQ','UCZHso3FEeHMv9bNRMbNF9_w']

def load_csv(file_path):
    with open(file_path) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(10240))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        return [row for row in reader if len(row)] #(i,name,author,pdf_urls)

def load_audio_meta(file_path):
    meta = {}
    with open(file_path,encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            videos = line['videos']
            durations = [v['duration'] for v in videos]
            durations.sort(reverse=True)
            min_duration = durations[0]*0.95
            videos = [v for v in videos if v['channel_id'] not in BLOCKED_CHANNELS and v['automatic_captions'] and v['duration']>=min_duration]
            key = line['query']
            if key.endswith(' audiobook'):
                key = key[:-len(' audiobook')]
            elif key.endswith(' audiobook female voice'):
                key = key[:-len(' audiobook female voice')]
            meta[key] = videos

    return meta

def pdf2text(src,target=None):
    if not target:
        target = os.path.splitext(src)[0] + '.txt'

    cmd = 'pdftotext "%s" "%s"'%(src,target)
    os.system(cmd)

    return target

def get_bookdir(root,name):
    name = re.sub('[^a-zA-Z.-0-9]','_',name)
    return os.path.join(root,name)

def download_first_pdf(urls,target):
    for url in urls:
        cmd = "wget -O --timeout=120'%s' '%s'"%(target,url)
        ret = os.system(cmd)
        if not ret:
            break

        # check if downloaded file is really pdf and log info about it

    if not os.path.isfile(target):
        raise Exception('unable to fetch: %s'%str(urls))

def download_audio_caption(vid,target_dir):
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
                                })
    with ydl:
        ydl.download([url])


def str2ms(time_str):

    m = re.match('([0-9]+):([0-9]+):([0-9]+).([0-9]+)',time_str)
    h,m,s,ms = m.group(1,2,3,4)
    return (int(h)*3600 + int(m)*60 + int(s))*1000+ int(ms)

'TODO'
def clean_text(text):

    return text

def split_audio(audio,caption,root,sr=22050,buffer=1800):
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
    for cap in tqdm(captions):
        # count += 1
        # if count >3: break
        if not cap['match']: continue
        tstart = time.time()
        text = cap['text']
        startstr = cap['start']
        endstr = cap['end']
        start = str2ms(startstr)
        end = str2ms(endstr)
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

    booklist = args.booklist
    audio_meta = args.audio_meta
    root = args.root_dir
    logfile = os.path.splitext(booklist)[0]+'.log.csv'
    logfile = open(logfile,'a')
    os.makedirs(root,exist_ok=True)
    booklist = load_csv(booklist)
    end = args.end if args.end >= 0 else len(booklist)
    booklist = booklist[args.start:end]
    audio_meta = load_audio_meta(audio_meta)
    logging.info('number of books: %s'%len(booklist))
    logfile.write('index|name|author|ispdf|istext|mp3|vtt2json|book_caption|numchunks|comments\n')

    for details in booklist:
        mstatus = Status()
        try:
            i,name,author,pdf_urls = details[:4]
            mstatus.i, mstatus.name, mstatus.author = i,name,author
            # logbook = '%s|%s|%s'%(i,name,author)
            key = '%s %s'%(name,author)
            videos = audio_meta.get(key,[])
            if not len(videos):
                # logfile.write(logfile+'-1|-1|-1')
                mstatus.comment += 'No video available'
                logfile.write(mstatus.tostr()+'\n')
                print('No videos available for %s'%key)
                continue

            bookdir = get_bookdir(root,name)
            os.makedirs(bookdir,exist_ok=True)
            pdf_urls = pdf_urls.split(' ')

            bookpath = os.path.join(bookdir,'book.pdf')
            if not os.path.isfile(bookpath):
                download_first_pdf(pdf_urls, bookpath )
            mstatus.ispdf = 1
            text_path = pdf2text(bookpath)
            mstatus.istext = 1

            for vid in videos:
                audio_path = os.path.join(bookdir,vid['id']+'.mp3')
                mono_audio_path = os.path.join(bookdir,vid['id']+'_mono.mp3')
                if not os.path.isfile(audio_path):
                    # download caption and audio
                    download_audio_caption(vid['id'],bookdir)
                if os.path.isfile(audio_path):
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
            isfirstline = True
            with open(meta_csv_path,'w') as f_meta_csv:
                for caption in captions:
                    try:
                        vid = os.path.basename(os.path.splitext(caption)[0])
                        new_caption = os.path.splitext(caption)[0]+'.json'
                        book_caption = os.path.splitext(caption)[0]+'_book.json'

                        if not os.path.isfile(new_caption):
                            new_caption = livevtt2json(caption,new_caption)
                        mstatus.vtt2json += '%s, '%vid

                        if not os.path.isfile(book_caption):
                            # align captions with book
                            book_caption = align_text(text_path,new_caption,book_caption)
                        mstatus.book_caption += '%s, '%vid

                        audio_path = book_caption[:-len('.en_book.json')]+'_mono.mp3'
                        print(book_caption,audio_path)
                        meta_csv = split_audio(audio_path,book_caption,root='%s/wavs'%bookdir,sr=22050,buffer=1800)
                        mstatus.numchunks += "%d, "%len(meta_csv)
                        meta_csv = '\n'.join(['|'.join(tpl) for tpl in meta_csv])
                        if isfirstline:
                            isfirstline = False
                        else:
                            meta_csv = '\n'+meta_csv
                        f_meta_csv.write(meta_csv)
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
    parser.add_argument('--audio_meta',required=True)
    parser.add_argument('--start',default=0,type=int,help='start index (including) of row to process(0 based)')
    parser.add_argument('--end',default=-1,type=int,help='last index (excluding) of row to process(counting based not the given in the list)')
    parser.add_argument('--root_dir',default='data',help='root directory path for all the data generated')
    args=parser.parse_args()
    run(args)
