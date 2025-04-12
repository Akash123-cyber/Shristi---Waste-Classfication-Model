import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from google.colab import drive

# Model Name
MODEL_NAME = "Shristi"

# Mount Google Drive
drive.mount('/content/drive')

# GPU Configuration
print(f"🚀 {MODEL_NAME} Waste Classification Model")
print("GPU Details:")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpu_devices}")

# Memory Management for T4
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14336)]
        )
        print("✅ T4 GPU Optimized Memory Configuration")
    except Exception as e:
        print(f"GPU Configuration Error: {e}")

# Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Dataset Path
dataset_path = '/content/images'  # Updated path for dataset

# Create directories for saving models and plots
os.makedirs('/content/drive/MyDrive/models', exist_ok=True)
os.makedirs('/content/drive/MyDrive/plots', exist_ok=True)

# Common Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Phase 1 Hyperparameters (Default Images)
PHASE1_LEARNING_RATE = 1e-4
PHASE1_EPOCHS = 30

# Phase 2 Hyperparameters (Real-world Images)
PHASE2_LEARNING_RATE = 1e-5
PHASE2_EPOCHS = 15

# Data Augmentation for Phase 1 (Default Images)
train_datagen_phase1 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Stronger Data Augmentation for Phase 2 (Real-world Images)
train_datagen_phase2 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

def create_data_generators(datagen, subset, phase):
    """Create data generators for both default and real-world images"""
    if phase == 1:
        # Phase 1: Use default images
        generator = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset=subset,
            shuffle=True,
            seed=42
        )
        return generator, None
    else:
        # Phase 2: Use real-world images
        generator = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset=subset,
            shuffle=True,
            seed=42
        )
        return None, generator

# Create Phase 1 Generators (Default Images)
train_default_gen, _ = create_data_generators(train_datagen_phase1, 'training', 1)
_, val_default_gen = create_data_generators(train_datagen_phase1, 'validation', 1)

print("\n🏷️ Phase 1 Class Mapping (Default Images):")
for class_name, class_index in train_default_gen.class_indices.items():
    print(f"{class_name}: {class_index}")

# Create Phase 2 Generators (Real-world Images)
train_real_gen, _ = create_data_generators(train_datagen_phase2, 'training', 2)
_, val_real_gen = create_data_generators(train_datagen_phase2, 'validation', 2)

def create_model(num_classes):
    """Create and compile the model"""
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax', dtype=tf.float32)
    ])
    
    return model, base_model

def create_callbacks(phase, learning_rate):
    """Create callbacks for training"""
    early_stopping = EarlyStopping(
        monitor='accuracy',
        patience=7 if phase == 1 else 5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='loss',
        factor=0.7,
        patience=4,
        min_lr=learning_rate/100,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        f'/content/drive/MyDrive/models/{MODEL_NAME}_phase{phase}_model.keras',
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, lr_reducer, model_checkpoint]

def plot_training_history(history, phase):
    """Plot training history for each phase"""
    plt.figure(figsize=(15,5))
    plt.suptitle(f'{MODEL_NAME} Phase {phase} Training Performance', fontsize=16)
    
    plt.subplot(1,3,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(history.history['precision_1'], label='Precision')
    plt.plot(history.history['val_precision_1'], label='Validation Precision')
    plt.plot(history.history['recall_1'], label='Recall')
    plt.plot(history.history['val_recall_1'], label='Validation Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/plots/{MODEL_NAME}_phase{phase}_performance.png')
    plt.close()

# Phase 1: Training on Default Images
print("\n🔄 Starting Phase 1: Training on Default Images")
model, base_model = create_model(train_default_gen.num_classes)
optimizer = Adam(learning_rate=PHASE1_LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

phase1_history = model.fit(
    train_default_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=val_default_gen,
    callbacks=create_callbacks(1, PHASE1_LEARNING_RATE)
)

# Phase 2: Fine-tuning on Real-world Images
print("\n🔄 Starting Phase 2: Fine-tuning on Real-world Images")

# Unfreeze the last 50 layers of the base model
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Recompile with lower learning rate
optimizer = Adam(learning_rate=PHASE2_LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

phase2_history = model.fit(
    train_real_gen,
    epochs=PHASE2_EPOCHS,
    validation_data=val_real_gen,
    callbacks=create_callbacks(2, PHASE2_LEARNING_RATE)
)

# Plot training histories
plot_training_history(phase1_history, 1)
plot_training_history(phase2_history, 2)

# Final Model Evaluation
print("\n🔍 Final Model Evaluation")
final_evaluation = model.evaluate(val_real_gen)
print(f"Final Validation Loss: {final_evaluation[0]:.4f}")
print(f"Final Validation Accuracy: {final_evaluation[1]*100:.2f}%")

# Save Final Model
model.save(f"/content/drive/MyDrive/models/{MODEL_NAME}_final_model.keras")
print(f"🚀 {MODEL_NAME} Two-Phase Training Complete! Models Saved Successfully!")
