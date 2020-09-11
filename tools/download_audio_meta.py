'''
python download_audio_meta.py --booklist goodreads/books_pdflog.tsv \
            --start 0 --end 2 --num_results 3 --nofemale
'''

import os, sys
import youtube_dl
from tqdm import tqdm
import csv
from termcolor import colored
import json
import time
import argparse

YT_KEY='AIzaSyCCyED1Pv6pGQgm1Qb-6HxoRo0m39OXdpA'

def download_idlist(file_path,vid=True,audio=True,caption=True):
    '''
    downloads all video ids mentioned in the list
    '''

    with open(file_path) as f:

        vids = f.readlines().split('\n')
        print('Found %d vids'%len(vids))
        ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})
        with ydl:

            for vid in tqdm(vids):
                vid = vid.strip()
                if not vid: continue
                url = 'http://www.youtube.com/watch?v='+vid
                ydl.download(['bestaudio[ext=m4a]' 'http://youtu.be/hTvJoYnpeRQ'])


def test_download_idlist():
    inp = sys.argv[1]
    download(inp,False,True,True)

def download_audio_meta(args):
    '''
    find audiobooks whose name and author are mentioned in the input file
    '''
    file_path = args.booklist
    end = args.end # actual number of rows in books where searching stopped
    with open(file_path) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(10240))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        search_items = []
        for i,row in enumerate(reader):
            if i < args.start or (args.end>0 and i>=args.end): continue
            if len(row) >= 3:
                end = i
                i,book,author = row[:3]
                # search_items.append('%s %s'%(book,author))
                if not args.nomale:
                    search_items.append('%s %s audiobook'%(book,author))
                if not args.nofemale:
                    search_items.append('%s %s audiobook female voice'%(book,author))
            else:
                print('Error: unexpected number of cols: %s'%str(row))

    opts = {
            'outtmpl': '%(title)s.%(ext)s',
            'write-auto-sub':True,
            'sub-lang':'en',
            'writesubtitles':True,
            'writeautomaticsub':True,
            'write-sub':True,
            'quiet':True,
            'no_warnings':True,
            'subtitleslangs':['en'],
            'ignoreerrors':True,
            'cookiefile':args.cookiefile
            }
    if args.sleep:
        opts['sleep_interval'] = 5
        opts['max_sleep_interval'] = 20


    ydl = youtube_dl.YoutubeDL(opts)

    meta_file = os.path.splitext(file_path)[0]+'_audio_meta_%d-%d.jsonl'%(args.start,end)
    id_cache = []
    if os.path.isfile(meta_file):
        mv_path = os.path.splitext(meta_file)[0]+'_%s.jsonl'%(time.asctime())
        print('Moving %s to %s'%(meta_file,mv_path))
        os.rename(meta_file,mv_path)
    with open(meta_file,'w',encoding='utf-8') as f_meta:
        with ydl:
            # searching
            keep_keys = ['id','webpage_url','uploader','channel_id','upload_date','title','subtitles','automatic_captions',
                            'duration','view_count','like_count','dislike_count','average_rating'
                          ]
            # num_books = 100
            num_results = args.num_results
            # print(colored('only using %d search items'%(2*num_books),'red'))
            # for item in tqdm(search_items[130:2*num_books]):
            for item in tqdm(search_items):
                try:
                    result = ydl.extract_info('ytsearch%d:%s'%(num_results, item), download=False)
                    if not result:continue
                    query = {'query':item}
                    videos = []
                    if 'entries' in result:
                        # Can be a playlist or a list of videos
                        # video = result['entries'][0]
                        for video in result['entries']:
                            id = video['id']
                            if id not in id_cache: id_cache.append(id)
                            else: continue
                            video = {k:video.get(k) for k in keep_keys}
                            video['automatic_captions'] = False if video['automatic_captions'].get('en') is None else True
                            video['subtitles'] = False if video['subtitles'].get('en') is  None else True
                            video['gender'] = ''
                            videos.append(video)
                            # print(json.dumps(video,indent=4))
                        query['videos'] = videos
                        f_meta.write(json.dumps(query,ensure_ascii=False)+'\n')
                    else:
                        # Just a video
                        # video = result
                        # raise Exception
                        print(str(result)[:500].replace('\n',' ') )
                        print('Error in %s'%item)
                        continue
                except Exception as e:
                    print(item,'\n',str(e))
                except KeyboardInterrupt:
                    break


if __name__=='__main__':
    # file_path = '/home/rajesh/work/limbo/data/yt/500books.tsv'
    # file_path = '/home/rajesh/work/limbo/data/yt/testbooks2.tsv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--booklist',required=True,help='path to booklist tsv')
    parser.add_argument('--start',default=0,type=int,help='start index (including) of row to process(0 based)')
    parser.add_argument('--end',default=-1,type=int,help='last index (excluding) of row to process(counting based not the given in the list)')
    parser.add_argument('--num_results',default=10,type=int,help='number of youtube videos to search')
    parser.add_argument('--nofemale',action='store_true',default=False,help='set to option to disable explicitly searching for female')
    parser.add_argument('--nomale',action='store_true',default=False,help='set to option to disable simple audiobook search (male search)')
    parser.add_argument('--sleep',action='store_true',default=False,help='enable sleeping')
    parser.add_argument('--cookiefile',default=None,help='path to cookiefile')
    args = parser.parse_args()
    download_audio_meta(args)
