import glob #Directory library
import logging 
import os
import cv2 #Computer vision
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm #Progress bar library

#Configuration
CONFIG = {
    "frame_dimensions": (720, 1280),
    "fps": 1,
    "batch_size": 32,
    "epochs": 50,
    "video_extension": "mp4",
    "policy1_dir": "./policy1",
    "policy2_dir": "./policy2",
    "training_video_name": "Bud_Light_15s_Folds_of_Honor_First_Responders_Fund.mp4", #Video selected for training
}


# ------------- SIAMESE ---------------#

# Siamese Network architecture
def siamese_network(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_siamese'),
    ])
    return model

# Data Preparation
def gather_video_files(directory, extension="mp4"):
    files = glob.glob(f"{directory}/*.{extension}")
    return files

# Frame Extraction
def extract_frames(video_file, target_size, fps):
    frames = []
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = round(frame_rate / fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
    cap.release()
    return frames

# Extract Features
def extract_siamese_features(model, frames):
    features = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        feature = model.predict(frame)
        features.append(feature.flatten())
    return np.array(features)

# Save Embeddings to CSV
def embeddings_to_csv(video_file, features, csv_filename):
    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    df.to_csv(csv_filename, index=False)


def main_siamese():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Gather video files for Policy 1 and Policy 2
    logging.info("Gathering video files for both policies.")
    policy1_files = gather_video_files(CONFIG["policy1_dir"], CONFIG["video_extension"])
    policy2_files = gather_video_files(CONFIG["policy2_dir"], CONFIG["video_extension"])

    # Check if video files are found
    if not policy1_files:
        logging.error("No video files found for Policy 1. Check the directory and file extension.")

    if not policy2_files:
        logging.error("No video files found for Policy 2. Check the directory and file extension.")


    # Load the Siamese network model
    logging.info("Loading Siamese network model.")
    siamese_model = siamese_network((1280, 720, 3))

    # Extract frames and features from the training video for pre-training
    logging.info("Extracting frames and features from the training video for pre-training.")
    training_video = os.path.join(CONFIG["policy1_dir"], CONFIG["training_video_name"])
    training_frames = [
        frame
        for frame in tqdm(extract_frames(training_video, CONFIG["frame_dimensions"], CONFIG["fps"]), desc="Training Video Frames")
    ]
    training_features = extract_siamese_features(siamese_model, training_frames)

    # Set the weights of the Siamese network based on features learned from the training video
    dense_siamese_weights = siamese_model.get_layer('dense_siamese').get_weights()
    siamese_model.layers[-1].set_weights(dense_siamese_weights)

    # Extract frames and features from each Policy 1 video and save to separate CSV files
    logging.info("Extracting frames and features from Policy 1 videos.")
    for policy1_file in tqdm(policy1_files, desc="Policy 1 Videos"):
        policy1_frames = [
            frame
            for frame in extract_frames(policy1_file, CONFIG["frame_dimensions"], CONFIG["fps"])]
        policy1_features = extract_siamese_features(siamese_model, policy1_frames)
        policy1_filename = os.path.basename(policy1_file)
        embeddings_to_csv([policy1_filename], policy1_features, f"policy1_{policy1_filename}_embeddings.csv")

    # Extract frames and features from Policy 2 videos and save to CSV
    logging.info("Extracting frames and features from Policy 2 videos.")
    policy2_frames = [frame for file in tqdm(policy2_files, desc="Policy 2 Videos") for frame in extract_frames(file, CONFIG["frame_dimensions"], CONFIG["fps"])]
    policy2_features = extract_siamese_features(siamese_model, policy2_frames)
    embeddings_to_csv(policy2_files, policy2_features, "policy2_embeddings.csv")

# ------------------ SimCLR --------------------------#


# SimCLR data augmentation
def get_simclr_augmentation(input_shape=(224, 224, 3)):
    data_augmentation = models.Sequential([
        preprocessing.Rescaling(1./255),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.5),
        preprocessing.RandomZoom(0.5),
        preprocessing.RandomContrast(0.5),
    ], name="simclr_augmentation")

    input_a = layers.Input(shape=input_shape)
    augmented_a = data_augmentation(input_a)

    return input_a, augmented_a

# SimCLR encoder
def get_simclr_encoder(base_encoder, projection_dim=128):
    inputs = layers.Input(shape=(None, None, 3))
    x = base_encoder(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(projection_dim, activation='relu')(x)
    encoder = models.Model(inputs, x)
    return encoder

# SimCLR model
def get_simclr_model(encoder, temperature=0.1):
    input_a, augmented_a = get_simclr_augmentation()

    # Encode augmented images
    encoded_a = encoder(augmented_a)

    # Create SimCLR model
    simclr_model = models.Model(inputs=input_a, outputs=encoded_a)

    return simclr_model


def extract_simclr_features(model, frames):
    features = []
    for frame in frames:
        # Resize frame to match the expected input shape of (224, 224)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, axis=0)
        feature = model.predict(frame)
        features.append(feature.flatten())
    return np.array(features)

def main_SimCLR():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Gather video files for Policy 1 and Policy 2
    logging.info("Gathering video files for both policies.")
    policy1_files = glob.glob(f"{CONFIG['policy1_dir']}/*.{CONFIG['video_extension']}")
    policy2_files = glob.glob(f"{CONFIG['policy2_dir']}/*.{CONFIG['video_extension']}")

    #Get base encoder
    base_encoder = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False)

    simclr_encoder = get_simclr_encoder(base_encoder)

    simclr_model = get_simclr_model(simclr_encoder)

    # Use the encoder and save embeddings to CSV for Policy 1 videos
    for policy1_file in tqdm(policy1_files, desc="Policy 1 Videos"):
        policy1_frames = [cv2.resize(frame, CONFIG["frame_dimensions"], interpolation=cv2.INTER_AREA) for frame in extract_frames(policy1_file, CONFIG["frame_dimensions"], CONFIG["fps"])]
        policy1_features = extract_simclr_features(simclr_model, policy1_frames)
        embeddings_to_csv([os.path.basename(policy1_file)], policy1_features, f"policy1_{os.path.basename(policy1_file)}_simclr_embeddings.csv")

    # Use the encoder and save embeddings to CSV for Policy 2
    for policy2_file in tqdm(policy2_files, desc="Policy 2 Videos"):
        policy2_frames = [cv2.resize(frame, CONFIG["frame_dimensions"], interpolation=cv2.INTER_AREA) for frame in extract_frames(policy2_file, CONFIG["frame_dimensions"], CONFIG["fps"])]
        policy2_features = extract_simclr_features(simclr_model, policy2_frames)
        embeddings_to_csv([os.path.basename(policy2_file)], policy2_features, f"policy2_{os.path.basename(policy2_file)}_simclr_embeddings.csv")

main_siamese()
main_SimCLR()