
############################### Imports ############################################
import tensorflow as tf
import keras
import shutil,os, cv2, gc
from glob import glob

import matplotlib.pyplot as plt
import numpy as np



from tqdm import tqdm
from sklearn import metrics
import itertools
import tensorflow_addons as tfa
from evaluation_metrics import *
import os
import warnings
warnings.filterwarnings('ignore')
from dataloader import ImageDataLoader
from your_model import eta_model
from baseline_models import CNN2D, InceptionV3, FCHCNN, PyFormer
############################### Hyperparameters ############################################
# Hyperparameters
data_dir  = '/home/pb/hb/shamas/disaster_classification_5_classes/AIDER'  # Dataset path
class_labels = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
# Define your model function names as strings
model_names = [CNN2D, InceptionV3, FCHCNN, PyFormer, eta_model]
SEED = 68765
input_shape = (224, 224, 3)
num_classes = len(os.listdir(data_dir))

# Train/Validation/Test splits
train_ratio = 0.70    # 70% for training
val_ratio = 0.15      # 15% for validation
test_ratio = 0.15     # 15% for testing

# Updated Hyperparameters
num_epochs = 5             # Increased to allow the model more training time
batch_size = 32
patch_size = 8               # Reduced to allow more fine-grained patch extraction
num_patches = (input_shape[0] // patch_size) ** 2
learning_rate = 0.0001      # Reduced for finer weight updates
weight_decay = 0.0001
label_smoothing = 0.01       # Reduced label smoothing
embedding_dim = 64          # Increased embedding dimension
mlp_dim = 64                # Increased MLP dimension for more complexity
dim_coefficient = 4
num_heads = 4                # Increased number of attention heads for better feature learning
attention_dropout = 0.2      # Increased dropout to prevent overfitting
projection_dropout = 0.2     # Increased dropout to prevent overfitting
num_transformer_blocks = 8  # Increased number of transformer blocks for better learning
tta_steps = 10
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")



############################### Output Directory ############################################
output_dir = f'results/epoches_{num_epochs}_batch_size_{batch_size}_train_{train_ratio*100}%'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

############################### GPU Configuration ############################################
# Set GPU device
def set_gpu_device(device_index):
     gpus = tf.config.list_physical_devices('GPU')
     if gpus:
         try:
            tf.config.experimental.set_visible_devices(gpus[device_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[device_index], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
         except RuntimeError as e:
            print(e)

# Call this function before any TensorFlow operations
set_gpu_device(1) # Set to use the second GPU (index 1)  #to change0 0.53125 0.484375 0.9375 0.875

############################### DataLoader Calling ############################################
data_loader = ImageDataLoader(
    data_dir,
    train_split=train_ratio,    # 70% for training
    val_split=val_ratio,     # 15% for validation
    test_split=test_ratio     # 15% for testing
)

# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()

print(f"Training set shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Number of classes: {data_loader.num_classes}")

############################### Baseline Models Calling ############################################
combined_results = {}
# Loop through the list of model function names
for model_name in model_names:
    print(f"Evaluating {model_name.__name__}")
    
    # Get the actual model instance by calling the function
    model = model_name(input_shape, num_classes)  
    
    # Save the model summary to a file
    file_name = f"{model_name.__name__}_summary.txt"
    with open(os.path.join(output_dir, file_name), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )  
    # Assuming `history` exists after model training
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_val, y_val)
    )


    # Save training performance plots
    save_training_plots(history, model_name.__name__, output_dir)
    
    # Save history as JSON
    save_history_to_json(history, model_name.__name__, output_dir)
    
    # Evaluate the model and save metrics
    metrics = evaluate_and_save_all_metrics(
        model, X_test, y_test, tta_steps, batch_size, class_labels, model_name.__name__, output_dir
    )

    # Store metrics in combined results
    combined_results[model_name.__name__] = metrics

# Save combined results to a JSON file
combined_results_path = os.path.join(output_dir, "combined_results.json")
with open(combined_results_path, 'w') as f:
    json.dump(combined_results, f, indent=4)

print(f"All metrics saved in {combined_results_path}")
