import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# Set paths
MODEL_PATH = '/home/akash/Shristi.keras'
TEST_DIR = '/home/akash/test'

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="husl")

# Custom color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6']

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Parameters (same as training)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Class mapping for reference
class_indices = test_generator.class_indices
class_names = list(class_indices.keys())
print("\nClass Mapping:")
for class_name, idx in class_indices.items():
    print(f"{class_name}: {idx}")

# Evaluate model
print("\nEvaluating model on test data...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Create output directory for plots
os.makedirs('/home/akash/Cursor/analysis_plots', exist_ok=True)

# 1. Enhanced Confusion Matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
plt.ylabel('True Category', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ROC Curves for each class
plt.figure(figsize=(12, 8))
y_bin = label_binarize(true_classes, classes=range(len(class_names)))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, 
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves for Each Class', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Class-wise Performance Metrics
plt.figure(figsize=(12, 8))
report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
class_metrics = pd.DataFrame(report).transpose()
class_metrics = class_metrics.iloc[:-3]  # Remove average and total rows

ax = class_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', color=colors[:3])
plt.title('Class-wise Performance Metrics', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics', bbox_to_anchor=(1.15, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/class_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Prediction Confidence Distribution
plt.figure(figsize=(12, 8))
confidence_scores = np.max(predictions, axis=1)
sns.histplot(confidence_scores, bins=30, color=colors[0], alpha=0.7)
plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Class-wise Sample Distribution
plt.figure(figsize=(12, 8))
class_counts = np.bincount(true_classes)
bars = plt.bar(class_names, class_counts, color=colors)
plt.title('Number of Samples per Class', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Top 3 Confusions
plt.figure(figsize=(12, 8))
confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
confusion_df = confusion_df.stack().reset_index()
confusion_df.columns = ['True', 'Predicted', 'Count']
confusion_df = confusion_df[confusion_df['True'] != confusion_df['Predicted']]
confusion_df = confusion_df.sort_values('Count', ascending=False).head(3)

sns.barplot(data=confusion_df, x='Count', y='True', hue='Predicted', palette=colors[:3])
plt.title('Top 3 Most Common Misclassifications', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
plt.ylabel('True Category', fontsize=12, fontweight='bold')
plt.legend(title='Predicted As', bbox_to_anchor=(1.15, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akash/Cursor/analysis_plots/top_confusions.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Print class-wise accuracy
print("\nClass-wise Accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask]) * 100
    print(f"{class_name}: {class_acc:.2f}%")

print("\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")

# Function to predict a single image
def predict_single_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    return class_names[predicted_class], confidence

# Example: Predict a few random test images
print("\nSample predictions:")
for category in class_names:
    category_dir = os.path.join(TEST_DIR, category)
    if os.path.exists(category_dir):
        files = os.listdir(category_dir)
        if files:
            sample_file = os.path.join(category_dir, files[0])
            predicted_category, confidence = predict_single_image(sample_file)
            print(f"Image: {sample_file}")
            print(f"True category: {category}")
            print(f"Predicted category: {predicted_category}")
            print(f"Confidence: {confidence:.2f}%")
            print("-" * 50)

print("\nDone!") 
