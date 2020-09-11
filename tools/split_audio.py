import sys,os
import librosa
import numpy as np
import wave

# sr=16000


def split(wav,start,end):
    global sr
    sw = 2
    ret = wav[int(start/1000*sr)*sw:int(end/1000*sr)*sw]
    return ret

if __name__ == '__main__':
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    wav_path = sys.argv[1]
    # wav = librosa.core.load(wav_path, sr=sr)[0]
    wav_file = wave.open(wav_path, 'rb')
    # wav = wav_file.readframes(wav_file.getnframes())
    sw=wav_file.getsampwidth()
    sr=wav_file.getframerate()
    channels = wav_file.getnchannels()

    startframe=int(start*sr/1000)
    numframe = int((end-start)*sr/1000)
    # clip = split(wav,start,end)
    wav_file.setpos(startframe)
    clip = wav_file.readframes(numframe)
    print('clipping from frame:%d to %d: expected %d'%(startframe,wav_file.tell(),numframe))

    clip_path,ext = os.path.splitext(wav_path)
    clip_path = clip_path +'_%d-%d'%(start,end) + ext
    print('Saving at: %s'%clip_path)

    audio = struct.unpack_from("%dh" %(len(clip)//sw), clip) # audio is of librosa format
    audio = np.array(audio)
    audio = audio / 32768.0
    librosa.output.write_wav(audio, wav.astype(np.int16), sr)
    # wave_file = wave.open(clip_path,'wb')
    # wave_file.setframerate(sr)
    # wave_file.setsampwidth(sw)
    # wave_file.setnchannels(channels)
    # wave_file.writeframes(clip)
    # wave_file.close()
