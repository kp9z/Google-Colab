import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import os
import tensorflow as tf
import datetime

def plot_loss_curve(model_history):
    """
    Plot loss curve of function & metrics on a seperate plot
    :param model_history: a
    :return: Plots of accuracy & metrics
    """
    df = pd.DataFrame(model_history.history)
    epochs = range(1,len(df)+1)

    plt.figure(figsize=(10,5))

    # Setup loss curve
    plt.subplot(121)
    plt.plot(epochs, df['loss'],label = 'Train loss')
    plt.plot(epochs, df['val_loss'], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    #Setup accuracy_curve
    plt.subplot(122)
    plt.plot(epochs, df['accuracy'],label = 'Train accuracy')
    plt.plot(epochs, df['val_accuracy'],label = 'Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Accuracy')

    plt.legend()
    plt.show()


def unzip_file(zip_url,unzip_dir = 'unzip_dir'):
    """
    Take in a zip URL, download and unzip file
    :param zip_url: (str) URL of the zip file
    :return: the path which unzip dir name/path
    """
    urllib.request.urlretrieve(zip_url, 'zipfile.zip')

    with zipfile.ZipFile('zipfile.zip', 'r') as zip_ref:
      zip_ref.extractall(unzip_dir)
    return unzip_dir

def walk_dir(root):
    """
    Walk through the directory, print out number of files & directory
    :param root: (str) Path of root URL to walk
    :return: print out number of dirs & files
    """
    for root,dirs, files in os.walk(root):
      print(f'There are {len(dirs)} directories & {(len(files))} files in {root}')


def create_tensorboard_callback(experiement_name,log_dir = 'tensor_board' ):
    """
    Create a TensorBoard callback
    :param log_dir: directory which log will be save
    :param experiement_name: (str) experiment name
    :return: a TensorFlow TensorBoard callback
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_full_path = log_dir+"/" + experiement_name+"/"+current_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_full_path)
    print(f'Saving log to {log_dir_full_path}')
    return tensorboard_callback

def combine_2_historys(history1,history2):
  """
  Combine 2 tensorflow history & display them
  """
  df1 = pd.DataFrame(history1.history)
  df2 = pd.DataFrame(history2.history)

  df = pd.concat([df1,df2],ignore_index=True)

  x = range(len(df))
  plt.figure(figsize = (10,7))
  plt.subplot(211)
  plt.plot(x,df.loss,label = 'loss')
  plt.plot(x,df.val_loss,label = 'val_loss')
  plt.vlines(len(df1)-1,0,max(df.loss))
  plt.legend()

  plt.subplot(212)
  plt.plot(x,df.accuracy,label = 'accuracy')
  plt.plot(x,df.val_accuracy,label = 'val_accuracy')
  plt.vlines(len(df1)-1,0,max(df.accuracy))

  plt.legend()

