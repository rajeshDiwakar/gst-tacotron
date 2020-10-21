'''
python split_audio.py \
  --audio jamendo/Avercage_-_Embers/Avercage_-_Embers_mono.wav \
  --caption jamendo/jsons/Avercage_-_Embers.json \
  --data_dir jamendo/Avercage_-_Embers/wavs \
  --silence 50,50
/home/rajesh/work/limbo/gst-tacotron/tools/jamendo/Avercage_-_Embers/Avercage_-_Embers.mp3
'''

import sys,os
import librosa
import numpy as np
import wave
import json
import logging
from tqdm import tqdm
import time
import re
import argparse
from pydub import AudioSegment

# logging.basicConfig(level=logging.DEBUG)

# sr=16000
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



def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms




def split_audio(audio,caption,root,sr=22050,buffer=1800,augmentation=None,aug_factor=3,min_duration=1,max_duration=10,silence='0'):
    '''
    buffer: audio chunk in seconds to load to avoid multiple mem reads
    '''
    with open(caption) as f:
        captions = json.load(f)
    os.makedirs(root,exist_ok=True)
    silence = silence.split(',')
    if len(silence)>= 2:
        lsilence,rsilence = int(silence[0]),int(silence[1])
    else:
        lsilence,rsilence=int(silence[0]),int(silence[0])
    lsilence=np.zeros(int(lsilence*sr//1000),dtype=np.int16)
    rsilence = np.zeros(int(rsilence*sr//1000),dtype=np.int16)

    meta=[]
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
        if not cap['match']: continue
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
        clip = np.concatenate((lsilence,clip,rsilence))
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

        # librosa.output.write_wav(clip_path, clip, sr)
        librosa.output.write_wav('temp.wav', clip, sr)

        # remove silence
        sound = AudioSegment.from_file('temp.wav', format="wav")
        margin = int(silence[0]) # in ms
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        start_trim = max(0,start_trim-margin)
        end_trim = max(0,end_trim-margin)

        duration = len(sound)
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export(clip_path,format='wav')

        # new_p = os.path.join('wavs',f)

        meta.append((clip_path,text,text))
    meta_csv_path = os.path.join(root,'../meta.csv')
    meta = ['|'.join(tpl) for tpl in meta]
    with open(meta_csv_path,'w',encoding='utf-8') as f:
                    f.write('\n'.join(meta))


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--audio',required=True,help='path to mp3 audio')
        parser.add_argument('--caption',required=True,help='path to json caption file')
        parser.add_argument('--data_dir',default='data',help='root directory path for all the data generated')
        parser.add_argument('--aug_factor',default=3,help='augmentation factor.')
        parser.add_argument('--augment',choices=['rand_step'],default=None,help='augmentation method')
        parser.add_argument('--min_duration',default=5,type=int)
        parser.add_argument('--max_duration',default=10,type=int)
        parser.add_argument('--silence',default='0',help='--silence=lsilence,rsilence in ms')

        args=parser.parse_args()
        split_audio(args.audio,args.caption,args.data_dir,augmentation=args.augment,aug_factor=args.aug_factor,
                    min_duration=args.min_duration,max_duration=args.max_duration,silence=args.silence)


#def split(wav,start,end):
#    global sr
#    sw = 2
#    ret = wav[int(start/1000*sr)*sw:int(end/1000*sr)*sw]
#    return ret

#if __name__ == '__main__':
#    start = int(sys.argv[2])
#    end = int(sys.argv[3])
#    wav_path = sys.argv[1]
#    # wav = librosa.core.load(wav_path, sr=sr)[0]
#    wav_file = wave.open(wav_path, 'rb')
#    # wav = wav_file.readframes(wav_file.getnframes())
#    sw=wav_file.getsampwidth()
#    sr=wav_file.getframerate()
#    channels = wav_file.getnchannels()

#    startframe=int(start*sr/1000)
#    numframe = int((end-start)*sr/1000)
#    # clip = split(wav,start,end)
#    wav_file.setpos(startframe)
#    clip = wav_file.readframes(numframe)
#    print('clipping from frame:%d to %d: expected %d'%(startframe,wav_file.tell(),numframe))

#    clip_path,ext = os.path.splitext(wav_path)
#    clip_path = clip_path +'_%d-%d'%(start,end) + ext
#    print('Saving at: %s'%clip_path)

#    audio = struct.unpack_from("%dh" %(len(clip)//sw), clip) # audio is of librosa format
#    audio = np.array(audio)
#    audio = audio / 32768.0
#    librosa.output.write_wav(audio, wav.astype(np.int16), sr)
#    # wave_file = wave.open(clip_path,'wb')
#    # wave_file.setframerate(sr)
#    # wave_file.setsampwidth(sw)
#    # wave_file.setnchannels(channels)
#    # wave_file.writeframes(clip)
#    # wave_file.close()
