import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def buildmodel():
    model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    i3d_model = hub.load(model_url).signatures['default']
    return i3d_model
print("model1 imported successfully")