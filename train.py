import argparse
from datetime import datetime
import math
import numpy as np
import os
import glob
import time
import subprocess
import time
import tensorflow as tf
import traceback
import textwrap

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
log = infolog.log


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if not os.path.isfile('mycreds.txt'):
    with open('mycreds.txt','w') as f:
        f.write('{"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-07-29T16:47:41Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "expires_in": 3599, "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}')

        # {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-08-09T09:46:00Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "expires_in": 3599, "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}


gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
# if gauth.credentials is None:
#     # Authenticate if they're not there
#     gauth.LocalWebserverAuth()
if gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

# drive = GoogleDrive(gauth)

def authorize_drive():
    # global drive
    global gauth
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    # if gauth.credentials is None:
    #     # Authenticate if they're not there
    #     gauth.LocalWebserverAuth()
    if gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)

    return drive


# def validate_parent_id(parent_id):
#     global drive
#     file_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
#         if len(file_list) > 1:
#             raise ValueError('There are multiple folders with that specified folder name')
#         elif len(file_list) == 0:
#             raise ValueError('No folders match that specified folder name')


def upload_to_drive(list_files,parent_id):
    # global drive
    drive = authorize_drive()
    # parent_id = ''# parent id
    for path in list_files:
        d,f = os.path.split(path)
        file = drive.CreateFile({'title': f, 'parents': [{'id': parent_id}]})
        file.SetContentFile(path)
        file.Upload()

def download_checkpoints(parent_id,root_dir='logs-tacotron'):
    drive = authorize_drive()
    downloaded_files = []
    os.makedirs(root_dir,exist_ok=True)
    # checkpoint = ''
    # file_list = drive.ListFile({'q': "title contains 'My Awesome File' and trashed=false"}).GetList()
    ckpt_path = os.path.join(root_dir,'checkpoint')
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()  #check if it is iterator
    # print(file_list)
    for f in file_list:
        if f['title'].lower() == 'checkpoint':
            file_id = f['id']
            file = drive.CreateFile({'id': file_id})
            file.GetContentFile(ckpt_path)
            downloaded_files.append(ckpt_path)
        elif f['title'].startswith('events'):
            file_id = f['id']
            file = drive.CreateFile({'id': file_id})
            file.GetContentFile(os.path.join(root_dir,f['title']))
            downloaded_files.append(os.path.join(root_dir,f['title']))

    if os.path.isfile(ckpt_path):
        with open(ckpt_path) as f:
            ckpt_data = f.read().split('\n')
        if len(ckpt_data):
            ckpt_data = ckpt_data[0].split(':')[-1].strip().strip('" ')
            weight_name = os.path.basename(ckpt_data)
            for f in file_list:
                if f['title'].startswith(weight_name):
                    file_id = f['id']
                    file = drive.CreateFile({'id': file_id})
                    file.GetContentFile(os.path.join(root_dir,f['title']))
                    downloaded_files.append(os.path.join(root_dir,f['title']))
    else:
        log('checkpoint file not found in drive')

    print('Downloaded following files\n%s'%'\n'.join(downloaded_files))



def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  parent_id = args.pid
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  if parent_id:
      log('Downloading model files from drive')
      download_checkpoints(parent_id)
  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

  # Train!
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = '%s |Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          time.asctime(), step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          list_files = [] #files to be uploaded to drive
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          prefix = saver.save(sess, checkpoint_path, global_step=step)
          list_files.extend(glob.glob(prefix+'.*'))
          log('Saving audio and alignment...')
          input_seq, spectrogram, alignment = sess.run([
            model.inputs[0], model.linear_outputs[0], model.alignments[0]])
          waveform = audio.inv_spectrogram(spectrogram.T)
          info = '\n'.join(textwrap.wrap( '%s, %s, %s, %s, step=%d, loss=%.5f' % (sequence_to_text(input_seq), args.model, commit, time_string(), step, loss),70, break_long_words=False) )
          audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
          plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
            info=info)
          log('Input: %s' % sequence_to_text(input_seq))

          list_files.append(os.path.join(log_dir, 'step-%d-audio.wav' % step))
          list_files.append(os.path.join(log_dir, 'step-%d-align.png' % step))
          if parent_id:
              try:
                  upload_to_drive(list_files,parent_id)
              except Exception as e:
                  print(e)
                  with open('drive_log.txt','a') as ferr:
                      ferr.write('\n\n\n'+time.asctime())
                      ferr.write('\n'+', '.join(list_files))
                      ferr.write('\n'+str(e))



    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.getcwd())
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--pid', default='', help='id of directory in google drive')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
