############################### Imports ############################################
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
class ImageDataLoader:
    def __init__(self, data_dir, img_height=224, img_width=224, 
                 train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Verify that splits sum to 1
        total_split = self.train_split + self.val_split + self.test_split
        if not np.isclose(total_split, 1.0):
            raise ValueError(f"Split ratios must sum to 1, got {total_split}")
        
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
    
    def _get_class_names(self):
        return sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])
    
    def _load_and_preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((self.img_width, self.img_height))
        return np.array(img) / 255.0
    
    def load_data(self):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_array = self._load_and_preprocess_image(img_path)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        X = np.array(images)
        y = np.array(labels)
        
        # One-hot encode the labels
        y = to_categorical(y, num_classes=self.num_classes)
        
        # First split: separate training set
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            train_size=self.train_split, 
            random_state=self.seed, 
            stratify=np.argmax(y, axis=1)  # Use original labels for stratification
        )
        
        # Second split: divide remaining data into validation and test sets
        val_ratio = self.val_split / (self.val_split + self.test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_ratio,
            random_state=self.seed,
            stratify=np.argmax(y_temp, axis=1)  # Use original labels for stratification
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_tf_dataset(self, batch_size=32):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(1000).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        
        return train_ds, val_ds, test_ds
    
