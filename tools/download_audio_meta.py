import os, sys
import youtube_dl
from tqdm import tqdm
import csv
from termcolor import colored
import json
import time

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

def download_audiobook_meta(file_path):
    '''
    find audiobooks whose name and author are mentioned in the input file
    '''
    with open(file_path) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        search_items = []
        for row in reader:
            if len(row) >= 3:
                i,book,author = row[:3]
                # search_items.append('%s %s'%(book,author))
                search_items.append('%s %s audiobook'%(book,author))
                search_items.append('%s %s audiobook female voice'%(book,author))
            else:
                print('Error: unexpected number of cols: %s'%str(row))

    ydl = youtube_dl.YoutubeDL({
                            'outtmpl': '%(title)s.%(ext)s',
                            'write-auto-sub':True,
                            'sub-lang':'en',
                            'writesubtitles':True,
                            'writeautomaticsub':True,
                            'write-sub':True,
                            'subtitleslangs':['en'],
                            'ignoreerrors':True
                                })

    meta_file = 'audiobook_meta.jsonl'
    id_cache = []
    if os.path.isfile(meta_file):
        mv_path = 'audiobook_meta-%s.bk.jsonl'%(time.asctime())
        print('Moving %s to %s'%(meta_file,mv_path))
        os.rename(meta_file,mv_path)
    with open(meta_file,'w',encoding='utf-8') as f_meta:
        with ydl:
            # searching
            keep_keys = ['id','webpage_url','channel_id','upload_date','title','subtitles','automatic_captions',
                            'duration','view_count','like_count','dislike_count','average_rating'
                          ]
            num_books = 100
            num_results = 10
            print(colored('only using %d search items'%(2*num_books),'red'))
            # for item in tqdm(search_items[130:2*num_books]):
            for item in tqdm(search_items[:2*num_books]):
                try:
                    result = ydl.extract_info('ytsearch%d:%s'%(num_results, item), download=False)
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
                        print(str(result)[:500] )
                        print('Error in %s'%item)
                        continue
                except Exception as e:
                    print(item,'\n',str(e))


if __name__=='__main__':
    # file_path = '/home/rajesh/work/limbo/data/yt/500books.tsv'
    # file_path = '/home/rajesh/work/limbo/data/yt/testbooks2.tsv'
    try:
        file_path = sys.argv[1]
    except Exception as e:
        print('python %s path/to/book.tsv'%(sys.argv[0]))
    download_audiobook_meta(file_path)
