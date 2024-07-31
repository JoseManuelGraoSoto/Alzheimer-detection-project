import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.utils import shuffle
import pandas as pd
import time

# Paths to your dataset
base_path = '/mnt/datosnas/josegrao/finalData/augmentationData/augPonderado'
res_path = '/mnt/datosnas/josegrao/finalData/augmentationData/resultadoAugMasDropOut'
res_pathM = '/mnt/datosnas/josegrao/finalData/augmentationData/resultadoAugMasDropOut/matricesAugTraining'

# Create the results directory if it does not exist
if not os.path.exists(res_path):
    os.makedirs(res_path)

# Create the matrices directory if it does not exist
if not os.path.exists(res_pathM):
    os.makedirs(res_pathM)

# Data directories
train_path = os.path.join(base_path, 'ENTRENAMIENTO')
val_path = os.path.join(base_path, 'VALIDACION')
test_path = os.path.join(base_path, 'TEST')

# Custom callback to print and save confusion matrix at the end of each epoch
class ConfusionMatrixCallback(Callback):
    def __init__(self, training_data, output_dir):
        super().__init__()
        self.training_data = training_data
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        train_generator = self.training_data
        train_steps = len(train_generator)
        y_train_true = []
        y_train_pred = []

        for step in range(train_steps):
            X_train, y_train = train_generator[step]
            y_train_true.extend(np.argmax(y_train, axis=1))
            y_train_pred.extend(np.argmax(self.model.predict(X_train, verbose=0), axis=1))  # Set verbose to 0

        conf_matrix = confusion_matrix(y_train_true, y_train_pred)
        print(f"Confusion matrix Aug at epoch {epoch + 1}:")
        print(conf_matrix)
        
        # Save confusion matrix as PNG
        plt.figure(figsize=(10, 7))
        plt.title(f'Confusion Matrix Aug at Epoch {epoch + 1}')
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'MCI', 'AD'], yticklabels=['CN', 'MCI', 'AD'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, f'matrizConfuAug-{epoch + 1}.png'))
        plt.close()

# DataGenerator class
class DataGenerator(Sequence):
    def __init__(self, filepaths, labels, batch_size=8, dim=(176, 176, 126), n_channels=1, n_classes=3, shuffle=True):
        self.filepaths = filepaths
        self.labels = np.array(labels, dtype=int)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_filepaths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_filepaths, batch_labels)
        if len(batch_filepaths) == 0 or len(batch_labels) == 0:
            raise ValueError(f"Empty batch at index {index}")
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.filepaths, self.labels = shuffle(self.filepaths, self.labels)

    def __data_generation(self, batch_filepaths, batch_labels):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.array(batch_labels, dtype=int)

        for i, filepath in enumerate(batch_filepaths):
            img = nib.load(filepath).get_fdata().astype(np.float32)
            img = np.expand_dims(img, axis=-1)
            img = self.normalize(img)
            X[i,] = img

        if np.any(y >= self.n_classes) or np.any(y < 0):
            raise ValueError(f"One or more labels are out of the valid range. Labels: {y}")

        y = to_categorical(y, num_classes=self.n_classes)

        return X, y

    def normalize(self, img):
        mean = np.mean(img)
        std = np.std(img)
        if std == 0:
            std = 1
        return (img - mean) / std  # Normalización Z-score

def load_filepaths_from_dir(directory, label):
    filepaths = []
    labels = []
    print(f"Loading file paths from {directory} and its subdirectories...")
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.nii'):
                filepath = os.path.join(root, filename)
                filepaths.append(filepath)
                labels.append(label)
    return filepaths, labels

train_ad_filepaths, train_ad_labels = load_filepaths_from_dir(os.path.join(train_path, 'AD'), 2)
train_cn_filepaths, train_cn_labels = load_filepaths_from_dir(os.path.join(train_path, 'CN'), 0)
train_mci_filepaths, train_mci_labels = load_filepaths_from_dir(os.path.join(train_path, 'MCI'), 1)

val_ad_filepaths, val_ad_labels = load_filepaths_from_dir(os.path.join(val_path, 'AD'), 2)
val_cn_filepaths, val_cn_labels = load_filepaths_from_dir(os.path.join(val_path, 'CN'), 0)
val_mci_filepaths, val_mci_labels = load_filepaths_from_dir(os.path.join(val_path, 'MCI'), 1)

test_ad_filepaths, test_ad_labels = load_filepaths_from_dir(os.path.join(test_path, 'AD'), 2)
test_cn_filepaths, test_cn_labels = load_filepaths_from_dir(os.path.join(test_path, 'CN'), 0)
test_mci_filepaths, test_mci_labels = load_filepaths_from_dir(os.path.join(test_path, 'MCI'), 1)

train_filepaths = train_ad_filepaths + train_cn_filepaths + train_mci_filepaths
train_labels = train_ad_labels + train_cn_labels + train_mci_labels

val_filepaths = val_ad_filepaths + val_cn_filepaths + val_mci_filepaths
val_labels = val_ad_labels + val_cn_labels + val_mci_labels

test_filepaths = test_ad_filepaths + test_cn_filepaths + test_mci_filepaths
test_labels = test_ad_labels + test_cn_labels + test_mci_labels

train_labels = np.array(train_labels, dtype=int)
val_labels = np.array(val_labels, dtype=int)
test_labels = np.array(test_labels, dtype=int)

print(f"Training set: {len(train_filepaths)} file paths")
print(f"Validation set: {len(val_filepaths)} file paths")
print(f"Test set: {len(test_filepaths)} file paths")

assert len(train_filepaths) == len(train_labels), "Inconsistent number of training file paths and labels"
assert len(val_filepaths) == len(val_labels), "Inconsistent number of validation file paths and labels"
assert len(test_filepaths) == len(test_labels), "Inconsistent number of test file paths and labels"

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv3D(8, (3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(layers.Dropout(0.7))  # 1prueba  0.6 
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(layers.Dropout(0.6))  # 1 prueba 0.4
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

input_shape = (176, 176, 126, 1)
print("Creating model...")
model = create_model(input_shape)
print("Model created.")
model.summary()

# Create data generators
train_generator = DataGenerator(train_filepaths, train_labels, batch_size=8)
val_generator = DataGenerator(val_filepaths, val_labels, batch_size=8)

print(f"Length of train generator: {len(train_generator)}")
print(f"Length of validation generator: {len(val_generator)}")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(os.path.join(res_path, 'best_modelPonderadoCroppedAug.h5'), monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-6)
conf_matrix_callback = ConfusionMatrixCallback(training_data=train_generator, output_dir=res_pathM)

# Start training with callbacks
print("Starting training...")
try:
    history = model.fit(train_generator, validation_data=val_generator, epochs=120, callbacks=[early_stopping, model_checkpoint, lr_scheduler, conf_matrix_callback], verbose=2)
except Exception as e:
    print(f"An error occurred: {e}")

print("Training completed.")

def save_plots(history, output_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plotPonderadoAug.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_plotPonderadoAug.png'))

    plt.show()

if 'history' in locals():
    save_plots(history, res_path)

    print("Evaluating model on test set...")
    test_generator = DataGenerator(test_filepaths, test_labels, batch_size=8, shuffle=False)

    # Verify the length of the test generator and predictions
    test_steps = len(test_generator)
    print(f"Length of test generator: {test_steps}")

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {test_accuracy}")

    print("Making predictions on test set...")
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    test_y_classes = np.concatenate([np.argmax(batch_y, axis=1) for _, batch_y in test_generator], axis=0)
    print(f"Length of y_pred_classes: {len(y_pred_classes)}")
    print(f"Length of test_y_classes: {len(test_y_classes)}")

    assert len(y_pred_classes) == len(test_y_classes), "Predictions and true labels have different lengths"

    print("Calculating precision, recall, and F1-score...")
    test_precision = precision_score(test_y_classes, y_pred_classes, average='weighted')
    test_recall = recall_score(test_y_classes, y_pred_classes, average='weighted')
    test_f1 = f1_score(test_y_classes, y_pred_classes, average='weighted')

    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test F1-Score: {test_f1}")

    conf_matrix = confusion_matrix(test_y_classes, y_pred_classes)
    class_report = classification_report(test_y_classes, y_pred_classes, target_names=['CN', 'MCI', 'AD'])

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'MCI', 'AD'], yticklabels=['CN', 'MCI', 'AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(res_path, 'confusion_matrixPonderadaAug.png'))
    plt.show()

    with open(os.path.join(res_path, 'classification_reportPonderadoAug.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write(class_report)

    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [test_accuracy, test_precision, test_recall, test_f1]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(res_path, 'test_metricsPonderadoAug.csv'), index=False)

    print("Metrics and report saved successfully.")
else:
    print("Training was not successful. No plots or reports were generated.")
