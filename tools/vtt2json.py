import re
import sys, os
import json
import logging
import webvtt

def livevtt2json(src_path,target_path =None):
    '''

00:17:37.039 --> 00:17:39.830 align:start position:0%
automatic pilot
often<00:17:37.440><c> in</c><00:17:37.600><c> a</c><00:17:37.679><c> very</c><00:17:38.000><c> advantageous</c><00:17:38.880><c> way</c><00:17:39.600><c> this</c>

00:17:39.830 --> 00:17:39.840 align:start position:0%
often in a very advantageous way this


00:17:39.840 --> 00:17:42.310 align:start position:0%
often in a very advantageous way this
creates<00:17:40.240><c> a</c><00:17:40.400><c> tendency</c><00:17:40.960><c> to</c><00:17:41.120><c> stick</c><00:17:41.360><c> to</c><00:17:41.520><c> routines</c>

00:17:42.310 --> 00:17:42.320 align:start position:0%
creates a tendency to stick to routines
    '''

    with open(src_path) as f:
        text = f.read()

    pat1='([0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)\s+-->\s+([0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)\s+align:start position:0%\n(.+)\n(.+)'
    pat2 = '<c>(.+?)</c><([0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)>'
    matches = re.finditer(pat1,text,re.MULTILINE)
    count = 0
    captions = []
    for match in matches:
        start, end, prev_text,curr_text = match.groups()
        # print(match.groups())
        if not curr_text.strip(): continue
        # if curr_text.find('<c>') <0:
        #     print('<EMPTY>'+curr_text+'</EMPTY>')
        #     continue

        curr_text = re.split('(<[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+>)',curr_text)
        curr_text[0] = '<c>%s</c>'%(curr_text[0])  # case: only one new word, two new word, three or more new word
        curr_text = '<%s>'%start + ''.join(curr_text) + '<%s>'%end
        submatches = re.finditer(pat2,curr_text)
        # print(curr_text)
        text = ''
        wstart = start
        caption = {"text":'',"start":start,"end":end,"words":[]}
        for submatch in submatches:
            w,wend = submatch.groups()
            text += w
            caption['words'].append({"text":w.strip(),"start":wstart,"end":wend})

            wstart = wend
        text = re.sub('\s+',' ',text)
        caption['text'] = text
        captions.append(caption)
        # only for extraction of sentence
        # curr_text  = re.sub('(<[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+>)|(<c>)|(</c>)','',curr_text)
        # curr_text = re.sub('\s+',' ',curr_text)
        # print(curr_text)

        # count += 1
        # if count > 10:
        #     break
    if not len(captions):
        raise ValueError('Unexpected format for live vtt:\n%s\n...'%text[:300])

    if not target_path:
        target_path = os.path.splitext(src_path)[0]+'.json'
        # print('writing output to %s'%out_path)
    with open(target_path,'w') as f:
        json.dump(captions, f,indent=4,ensure_ascii=False)

    return target_path


if __name__ == '__main__':
    if len(sys.argv)== 1:sys.argv.append('/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai Audiobook full _ Hector Garcia and Francc Miralles.en.vtt')
    print(livevtt2json(sys.argv[1]))
