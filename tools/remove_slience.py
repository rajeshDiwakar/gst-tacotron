import os, sys
from pydub import AudioSegment

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

if __name__ == '__main__':

        inp_file = sys.argv[1]
        out_file = os.path.splitext(inp_file)[0]+'_clean.wav'
        sound = AudioSegment.from_file(inp_file, format="wav")
        
        margin = 10 # in ms
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        start_trim = max(0,start_trim-margin)
        end_trim = max(0,end_trim-margin)
        
        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export(out_file,format='wav')
