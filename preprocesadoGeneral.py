import os
import numpy as np
import SimpleITK as sitk
from skimage import filters

def normalize_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def segment_image(img):
    otsu_threshold = filters.threshold_otsu(img)
    binary_img = img > otsu_threshold
    return binary_img.astype(np.float32)

def preprocess_image(input_path, output_folder):
    # Cargar la imagen usando SimpleITK
    img = sitk.ReadImage(input_path)
    img_array = sitk.GetArrayFromImage(img).astype(np.float32)
    
    # Normalización
    img_normalized = normalize_image(img_array)
    
    # Segmentación
    img_segmented = segment_image(img_normalized)
    
    # Verificación de dimensiones
    if img_array.shape != img_segmented.shape:
        print(f"Dimensiones incorrectas: {img_segmented.shape}, deberían ser {img_array.shape}")
    else:
        print(f"Dimensiones correctas: {img_segmented.shape}")

    # Construir el nombre del archivo de salida
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_name = f"{name}_PREPROCESADO{ext}"
    output_path = os.path.join(output_folder, output_name)
    
    # Guardar la imagen preprocesada usando SimpleITK
    segmented_img_sitk = sitk.GetImageFromArray(img_segmented)
    segmented_img_sitk.CopyInformation(img)
    sitk.WriteImage(segmented_img_sitk, output_path)
    print(f"Imagen guardada en: {output_path}")

def preprocess_dataset(data_folder, output_folder):
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".nii"):
                input_path = os.path.join(root, file)
                # Construir la ruta de salida correspondiente
                relative_path = os.path.relpath(root, data_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                preprocess_image(input_path, output_subfolder)

# Ruta del conjunto de datos
data_folder = 'ruta entrada imaganes para aplicar preprocesado'
output_folder = 'ruta salida imaganes para aplicar preprocesado'

preprocess_dataset(data_folder, output_folder)
