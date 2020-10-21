'''
https://search.azlyrics.com/search.php?q=White+Flag+dido
https://www.azlyrics.com/lyrics/dido/whiteflag.html
'''
from bs4 import BeautifulSoup as bsp
import requests
import os,sys,json,re
import traceback
from tqdm import tqdm
import time

class Res():
    def __init__(self):
        self.status_code = 200
        with open('/home/rajesh/work/limbo/gst-tacotron/tools/att') as f:
            self.text = f.read()


def get_lyric_url(title='attention',singer='charlie puth'):
    res = requests.get('https://search.azlyrics.com/search.php',params={'q':'%s %s'%(title,singer)})
    title_words = [m.group() for m in re.finditer('[a-zA-Z][a-zA-Z]+',title.lower())]
    singer_words = [m.group() for m in re.finditer('[a-zA-Z][a-zA-Z]+',singer.lower())]
    query = set(title_words+singer_words)
    for i in range(1):
        if res.status_code == 200:
            soup = bsp(res.text)
            smalls = soup.find_all('small')
            for s in smalls:s.extract()
            rows = soup.find_all('td',{'class':'text-left'})
            # search_lists = soup.find_all('small',{"class":'search-list'})
            match_search_list = None
            best_dist = 1000
            best_match = set()
            for row in rows:
                # print(search_list.get_text())
                title = row.get_text()
                artist = ''#search_list.find('a',{"class":'artist'}).get_text()
                # print(title,artist)
                # candidate = set(title.lower().split(' ')+artist.lower().split(' '))
                candidate = set([m.group() for m in re.finditer('[a-zA-Z][a-zA-Z]+',title.lower())] +
                                [m.group() for m in re.finditer('[a-zA-Z][a-zA-Z]+',artist.lower())]
                                )
                dist = len(query)-len(query.intersection(candidate))
                if dist==0:
                    urls = row.findChildren('a',recursive=True)
                    urls=[a['href'] for a in urls]
                    # print(urls)
                    urls = [url for url in urls if url.startswith('https://www.azlyrics.com/lyrics')]
                    url = urls[0]
                    # url = 'https://thelyricsearch.com'+url
                    return url,title,artist
                else:
                    if dist< best_dist:
                        best_dist = dist
                        best_match = candidate

            print('No matching singer+title found')
            print(best_dist,best_match)
            break
        elif res.status_code == 502:
            print('server overloaded')
            time.sleep(180)
        else:
            print('Song search returned not ok status code',res.status_code)
    return None,None,None

# def get_page():
#
#     requests.get(url)

def get_lyrics(url):

    res = requests.get(url)
    lyrics = None
    for i in range(1):
        if res.status_code == 200:
            text = res.text
            soup = bsp(text)
            match = soup.find_all('div',{'class':'ringtone'})
            if len(match):
                nextNode = match[0]
                while True:
                    nextNode = nextNode.nextSibling
                    try:
                        tag_name = nextNode.name
                    except AttributeError:
                        tag_name = ""
                    if tag_name == "div":
                        brs = nextNode.find_all('br')
                        if len(brs)>= 3:
                            return nextNode.get_text(separator=u"\n")

            else:
                print('no class= ringtone found')
            return lyrics
        elif res.status_code == 502:
            print('server overloaded')
            time.sleep(180)
        else:
            print('Not Ok status code (=%d) for: %s'%(res.status_code,url))
    return lyrics

os.makedirs('cache',exist_ok=True)
os.makedirs('lyricsaz',exist_ok=True)

def main():
    with open('/home/rajesh/Music/oth/boyce/playlist1/songs.json') as f:
        songs = json.load(f)
    log = open('logaz.tsv','a')
    for song in tqdm(songs):
        try:
            title = song['title']
            singer = song['singer']
            vid = song['vid']
            lyric_path = 'lyricsaz/%s.txt'%vid
            if  os.path.isfile(lyric_path):
                print('%s already exists'%lyric_path)
                continue
            url,true_title,true_artist = get_lyric_url(title,singer)
            # log.write('%s\t%s\t%s\t%s\n'%(song['name'],true_title,true_artist,url))
            if url is None:
                print('failed to extract url for: ',title,singer,vid)
                continue
            else:
                print('lyric url: ',url)
            time.sleep(2)

            # continue
            lyric = get_lyrics(url)
            if lyric is None:
                print('failed to get lyrics from : %s for '%url,title,singer,vid)
            else:
                with open(lyric_path,'w') as f:
                    f.write(lyric)
        except Exception as e:
            # print(e)
            traceback.print_exc()
            continue
        except KeyboardInterrupt:
            log.close()
            break


# get_lyric_url()
main()
