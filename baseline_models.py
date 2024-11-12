############################### Imports ############################################
import tensorflow as tf
from tensorflow.keras import layers,models, regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense,  Dropout, concatenate
from tensorflow.keras.models import Model

def CNN2D(input_shape, num_classes):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # First Conv2D + MaxPooling2D block
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Second Conv2D + MaxPooling2D block
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Third Conv2D + MaxPooling2D block
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Flatten the feature map
    flatten = Flatten()(pool3)
    
    # Fully connected layer
    dense1 = Dense(512, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dense1)
    
    # Define the Model with Input and Output layers
    model = Model(inputs=input_layer, outputs=output_layer, name='CNN2D')
    
    return model

def InceptionV3(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Initial Conv and MaxPool layers
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='valid')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(80, (1, 1), activation='relu', padding='valid')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='valid')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # First Inception Module
    branch1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    branch5x5 = Conv2D(48, (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch5x5)

    branch3x3dbl = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    branch3x3dbl = Conv2D(96, (3, 3), padding='same', activation='relu')(branch3x3dbl)
    branch3x3dbl = Conv2D(96, (3, 3), padding='same', activation='relu')(branch3x3dbl)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(32, (1, 1), padding='same', activation='relu')(branch_pool)

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

    # Second Inception Module
    branch1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    branch5x5 = Conv2D(48, (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(branch5x5)

    branch3x3dbl = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    branch3x3dbl = Conv2D(96, (3, 3), padding='same', activation='relu')(branch3x3dbl)
    branch3x3dbl = Conv2D(96, (3, 3), padding='same', activation='relu')(branch3x3dbl)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(64, (1, 1), padding='same', activation='relu')(branch_pool)

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

    # Global Average Pooling and Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)

    # Output layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs, name='InceptionV3')

    return model

def FCHCNN(input_shape, num_classes):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Conv2D layers to replace Conv3D for RGB images
    conv_layer1 = Conv2D(filters=8, kernel_size=(7, 7), activation='relu')(input_layer)
    conv_layer2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_layer2)
    
    # If you want to add another Conv2D layer
    # conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)
    
    # Flattening before fully connected layers
    flatten_layer = Flatten()(conv_layer3)
    
    # Fully connected layers (Dense layers)
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    
    # Output layer for classification
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer2)
    
    # Define the Model with Input and Output layers
    model = Model(inputs=input_layer, outputs=output_layer, name='FCHCNN')
    
    return model

# Define the Pyramid CNN model (2D version for RGB images)
def pyramid_cnn(input_shape, num_pyramid_levels):
    # input_shape = (WS, WS, k)
    max_patch_size = min(input_shape[:2])
    patch_size = max_patch_size
    inputs = layers.Input(shape=input_shape)
    x = inputs
    # Create pyramid levels
    pyramid_outputs = []
    
    for level in range(num_pyramid_levels):
        scale = 2 ** level
        scaled_input_shape = (input_shape[0] // scale, input_shape[1] // scale, input_shape[2])
        
        # Convolutional layers for each pyramid level (2D Convolutions)
        x = layers.Conv2D(32, kernel_size=(patch_size, patch_size), activation='relu', padding='same', input_shape=scaled_input_shape)(x)
        x = layers.Conv2D(64, kernel_size=(patch_size, patch_size), activation='relu', padding='same')(x)
        pyramid_outputs.append(x)

    # Merge the feature maps from pyramid levels
    merged = layers.concatenate(pyramid_outputs, axis=-1)
    return Model(inputs=inputs, outputs=merged)
def PyFormer(input_shape, num_classes):
    num_layers = 2
    num_heads = 4
    mlp_dim = 64
    num_pyramid_levels = 2
    k = 3  # number of channels for RGB
    
    inputs = layers.Input(shape=input_shape)
    
    # Use the 2D Pyramid CNN model
    x = pyramid_cnn(input_shape, num_pyramid_levels)(inputs)
    x = layers.Activation("relu")(x)
    
    for i in range(num_layers):
        x = layers.LayerNormalization()(x)
        
        # Apply MultiHeadAttention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=k)(x, x)
        
        # Project 'inputs' to match the number of channels in 'attn_output' before adding
        projected_inputs = layers.Conv2D(filters=attn_output.shape[-1], kernel_size=(1, 1))(inputs)
        
        # Add attention output to the projected inputs
        x = layers.Add()([attn_output, projected_inputs])
        
        # 2D Conv layers instead of 3D
        y = layers.Conv2D(filters=k, kernel_size=(3, 3), activation='relu', padding='same')(x)
        y = layers.Conv2D(filters=2 * mlp_dim, kernel_size=(3, 3), activation='relu', padding='same')(y)
        x = layers.Add()([x, y])
    
    # Apply regularization and final dense layers
    flatten_layer = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(flatten_layer)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='PyFormer')
    return model

