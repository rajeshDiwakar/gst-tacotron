'''

takes a caption-json and audio file and extracts style for each caption
if you have audio and aligned book_caption
use this script to get style from audio for the captions. It will used to train caption2style model

checkpoint="/home/rajesh/work/limbo/gst-tacotron/logs-tacotron/model.ckpt-29000"
text="-"
captions='/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai.en_book.json'
ref_audio='/home/rajesh/work/limbo/data/yt/audiobooks/Ikigai_mono.wav'
python extract_style.py --checkpoint "$checkpoint" --text "$text" --reference_audio "$ref_audio" --captions "$captions"

python extract_style.py --checkpoint "$checkpoint" --hparams="max_iters=1500,batch_size=16" --text "$text" --reference_audio "$ref_audio" --captions "$captions"

'''

import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio
import time
from tqdm import tqdm
import json
import wave
import librosa
from base64 import b64encode

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)

def str2ms(time_str):

    m = re.match('([0-9]+):([0-9]+):([0-9]+).([0-9]+)',time_str)
    h,m,s,ms = m.group(1,2,3,4)
    return (int(h)*3600 + int(m)*60 + int(s))*1000+ int(ms)


def run_eval(args):
  print(hparams_debug_string())
  is_teacher_force = False
  mel_targets = args.mel_targets
  reference_mel = None
  if args.mel_targets is not None:
    is_teacher_force = True
    mel_targets = np.load(args.mel_targets)
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.checkpoint, args.reference_audio, style_path=args.style)
  base_path = get_output_base_path(args.reference_audio)
  ref = 'audio' if args.reference_audio else 'style'

  reference_audio = None
  style_path = None
  mel_target = None

  with open(args.captions) as f:
      captions = json.load(f)



  new_captions = []
  count = 0
  for cap in tqdm(captions[100:150]):
      count += 1
      if count >3: break
      if not cap['match']: continue
      tstart = time.time()
      text = cap['text']
      start = cap['start']
      end = cap['end']
      # reference_audio = input('Ref Audio: ').strip().strip("'")
      # ref_wav = audio.load_wav(reference_audio)
      duration = (str2ms(end) - str2ms(start))/1000.0
      ref_wav,_ = librosa.core.load(args.reference_audio,offset=str2ms(start)/1000.0, duration=duration)
      reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
      # path = '%s_ref-%s.wav' % (base_path, os.path.splitext(os.path.basename(reference_audio))[0])
      alignment_path = '%s_ref_%s-%s_align.png' % (base_path, start,end)
      new_audio_path = '%s_ref_%s-%s.wav' % (base_path, start,end)
      # style_path = '%s_ref-%s-style' % (base_path, os.path.splitext(os.path.basename(reference_audio))[0])
      # print('Output style: %s' % style_path)

      # elif ref == 'style':
      #     reference_mel = None
      #     style_path = input('Style Path: ').strip().strip("'")
      #     path = '%s_ref-%s.wav' % (base_path, os.path.splitext(os.path.basename(style_path))[0])
      #     alignment_path = '%s_ref-%s-align.png' % (base_path, os.path.splitext(os.path.basename(style_path))[0])
      # # if not style_path and reference_audio:
      # #     print('Atleast one of ref audio or style path must be  specified')

      with open(new_audio_path, 'wb') as new_audio:
        # print('Synthesizing: %s' % args.text)
        # print('Output wav file: %s' % path)
        print('Output alignments: %s' % alignment_path)

        new_audio.write(synth.synthesize(text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path, style_path=None))
        # synth.synthesize(text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path, style_path=None)
        cap['style'] = b64encode(synth.style.tostring()).decode('ascii')
        new_captions.append(cap)
        print('Took: %f s'%(time.time()-tstart))

  new_cap_path = os.path.splitext(args.captions)[0]+'_style'+'.json'
  with open(new_cap_path,'w') as f:
      json.dump(new_captions,f,indent=4)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference_audio', default=None, help='Reference audio path')
  parser.add_argument('--style', default=None, help='style embedding path')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  parser.add_argument('--captions', required=True, help='json file containing the book aligned captions')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
