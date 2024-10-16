import glob
import pandas as pd
import cv2
import gc
import numpy as np
import random
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from urllib.request import urlopen
from keras.models import load_model
import tensorflow_hub as hub


class CFG:
    epochs = 5
    batch_size = 32
    classes = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
        "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
        "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
        "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
        "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
        "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow",
        "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
        "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
        "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
        "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
        "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
        "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
        "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
        "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
        "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
        "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
        "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
        "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
        "YoYo"
    ]
    videos_per_class = 10

def build_model():
    def format_frames(frame, output_size):
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame

    def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
    # Read each video frame by frame
        result = []
        src = cv2.VideoCapture(str(video_path))  

        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.randint(0, max_start + 1)

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = src.read()
        result.append(format_frames(frame, output_size))

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
        src.release()
        result = np.array(result)[..., [2, 1, 0]]
        return result

    def to_gif(images):
        converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
        imageio.mimsave('./animation.gif', converted_images, duration=0.1)

    def classify_video_url(model, video_url, n_frames=10):
        # Download video from URL
        video = urlopen(video_url)
        with open('temp_video.avi', 'wb') as f:
            f.write(video.read())

        # Create frames from the downloaded video
        video_frames = frames_from_video_file('temp_video.avi', n_frames=n_frames)

        # Predict using the model
        predictions = model.predict(np.expand_dims(video_frames, axis=0))

        # Classify the video based on predictions
        predicted_class = np.argmax(predictions)
        predicted_class_name = CFG.classes[predicted_class]

        print(f"Predicted Class: {predicted_class_name}")
        return predicted_class_name

    # Load UCF101 dataset
    file_paths = []
    targets = []
    for i, cls in enumerate(CFG.classes):
        sub_file_paths = glob.glob(f"/kaggle/input/ucf101/UCF101/UCF-101/{cls}/**.avi")[:CFG.videos_per_class]
        file_paths += sub_file_paths
        targets += [i] * len(sub_file_paths)

    # Create features
    features = []
    for file_path in tqdm(file_paths):
        features.append(frames_from_video_file(file_path, n_frames=10))
    features = np.array(features)

    # Split dataset
    train_features, val_features, train_targets, val_targets = train_test_split(features, targets, test_size=0.2, random_state=42)


    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)).shuffle(CFG.batch_size * 4).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)
    valid_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # Build model
    net = tf.keras.applications.EfficientNetB0(include_top=False)
    net.trainable = False

    efficient_net_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(255.0),
        tf.keras.layers.TimeDistributed(net),
        tf.keras.layers.Dense(len(CFG.classes)),
        tf.keras.layers.GlobalAveragePooling3D()
    ])

    efficient_net_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train model
    history = efficient_net_model.fit(
        train_ds,
        epochs=CFG.epochs,
        validation_data=valid_ds,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(
                "efficient_net_model.h5",
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True
            )
        ]
    )

# Load the model from the .keras file
# model1 = load_model('model111.keras')
model1 = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']