"""
Soil Image Classification Training Script
Trains a Convolutional Neural Network (Transfer Learning via MobileNetV2 or Custom CNN)
on the 7 Soil classes:
- Alluvial_Soil
- Arid_Soil
- Black_Soil
- Laterite_Soil
- Mountain_Soil
- Red_Soil
- Yellow_Soil
"""

import os
import json
import numpy as np

def train_model(
    dataset_dir="Orignal-Dataset/Orignal-Dataset", 
    output_model_path="backend/models/soil_classifier.keras",
    classes_json_path="backend/models/soil_classes.json",
    img_size=(224, 224), 
    batch_size=32, 
    epochs=12
):
    if not os.path.exists(dataset_dir):
        if os.path.exists("Orignal-Dataset") and os.path.exists("Orignal-Dataset/Alluvial_Soil"):
            dataset_dir = "Orignal-Dataset"
        elif os.path.exists("../Orignal-Dataset/Orignal-Dataset"):
            dataset_dir = "../Orignal-Dataset/Orignal-Dataset"
            output_model_path = "../" + output_model_path
            classes_json_path = "../" + classes_json_path

    print("Loading Soil Dataset from:", dataset_dir)
    
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("TensorFlow is not yet installed. Run pip install -r requirements.txt first.")
        return

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    print("Loading Training Generator...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    print("Loading Validation Generator...")
    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    class_indices = train_generator.class_indices
    index_to_class = {str(v): k for k, v in class_indices.items()}
    print("Detected Soil Classes:", class_indices)

    with open(classes_json_path, "w") as f:
        json.dump(index_to_class, f, indent=4)
    print("Class mappings saved to:", classes_json_path)

    num_classes = len(class_indices)

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training Soil CNN model...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    print("Saving model to:", output_model_path)
    model.save(output_model_path)
    print("Soil CNN training complete!")

if __name__ == "__main__":
    train_model()
