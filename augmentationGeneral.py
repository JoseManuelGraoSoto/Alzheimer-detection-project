import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import shutil
import random

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Añadir sal
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 1

    # Añadir pimienta
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image

def adjust_brightness(img):
    gain = np.random.uniform(0.9, 1.1)
    return gain * img

def flip_imageX(img):
    if np.random.rand() > 0.5:
        return np.flip(img, axis=0)
    return img

def flip_imageY(img):
    if np.random.rand() > 0.5:
        return np.flip(img, axis=1)
    return img

def flip_imageZ(img):
    if np.random.rand() > 0.5:
        return np.flip(img, axis=2)
    return img

def elastic_transform(image, alpha, sigma):
    image = sitk.GetImageFromArray(image)
    transform = sitk.BSplineTransformInitializer(image, [4, 4, 4])
    params = np.asarray(transform.GetParameters(), dtype=float)
    params = params + np.random.randn(params.shape[0]) * sigma
    transform.SetParameters(tuple(params))
    return sitk.GetArrayFromImage(sitk.Resample(image, transform))

def zoom_image(image, zoom_range=(0.9, 1.1)):
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    image_sitk = sitk.GetImageFromArray(image)

    # Calcular el nuevo tamaño
    original_size = np.array(image_sitk.GetSize(), dtype=int)
    new_size = (original_size * zoom_factor).astype(int).tolist()

    # Calcular el nuevo espaciado
    original_spacing = np.array(image_sitk.GetSpacing())
    new_spacing = original_spacing / zoom_factor

    # Crear la imagen resampleada
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_sitk)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampled_image = resampler.Execute(image_sitk)

    # Asegurar que la imagen tenga la misma forma que la original
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    original_shape = image.shape
    zoomed_shape = resampled_array.shape

    cropped_padded_image = np.zeros_like(image)

    start = [(o - z) // 2 if o > z else 0 for o, z in zip(original_shape, zoomed_shape)]
    end = [start[i] + min(original_shape[i], zoomed_shape[i]) for i in range(3)]

    start_zoom = [(z - o) // 2 if z > o else 0 for z, o in zip(zoomed_shape, original_shape)]
    end_zoom = [start_zoom[i] + min(original_shape[i], zoomed_shape[i]) for i in range(3)]

    cropped_padded_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = \
        resampled_array[start_zoom[0]:end_zoom[0], start_zoom[1]:end_zoom[1], start_zoom[2]:end_zoom[2]]

    return cropped_padded_image

def translate_image(image, shift_range=4):
    translation = [np.random.uniform(-shift_range, shift_range) for _ in range(3)]
    transform = sitk.TranslationTransform(3, translation)
    return sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(image), transform))

def rotate_image(image, angle_range=5):
    angles = [np.random.uniform(-angle_range, angle_range) for _ in range(3)]
    transform = sitk.Euler3DTransform()
    transform.SetRotation(np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2]))
    return sitk.GetArrayFromImage(sitk.Resample(sitk.GetImageFromArray(image), transform))


def augment_image(input_path, output_folder, num_augmented_images=8):
    img = nib.load(input_path).get_fdata().astype(np.float32)
    
    for i in range(num_augmented_images):
        # Añadir ruido Salt & Pepper
        img_noisy = add_salt_and_pepper_noise(img)
                
        # Ajustar brillo
        img_brightness = adjust_brightness(img_noisy)
        
        # Seleccionar aleatoriamente una transformación
        img_transformed1 = random.choice([
            flip_imageX, flip_imageY, flip_imageZ,
        ])(img_brightness)
        
        img_transformed = random.choice([
            lambda x: translate_image(x, shift_range=3),
            lambda x: rotate_image(x, angle_range=5)
        ])(img_transformed1)
        
        if i % 2 == 0:
            img_zoom = zoom_image(img_transformed, zoom_range=(0.9, 1.1))
        
        # Aplicar transformación elástica
        img_elastic = elastic_transform(img_zoom, alpha=1, sigma=0.4)

        # Verificación de dimensiones
        if img_elastic.shape != img.shape:
            print(f"Dimensiones incorrectas: {img_elastic.shape}, deberían ser {img.shape}")
        else:
            print(f"Dimensiones correctas: {img_elastic.shape}")

        # Construir el nombre del archivo de salida
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        output_name = f"{name}_AUG_{i + 1}{ext}"
        output_path = os.path.join(output_folder, output_name)
        
        # Guardar la imagen aumentada
        nib.save(nib.Nifti1Image(img_elastic, np.eye(4)), output_path)
        print(f"Imagen guardada en: {output_path}")

def augment_dataset(data_folder, output_folder, num_augmented_images=8):
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".nii"):
                input_path = os.path.join(root, file)
                # Construir la ruta de salida correspondiente
                relative_path = os.path.relpath(root, data_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                
                # Si la carpeta es de VALIDACION o TEST, solo copiar las imágenes
                if 'VALIDACION' in root or 'TEST' in root:
                    shutil.copy(input_path, output_subfolder)
                    print(f"Imagen copiada sin modificaciones: {file}")
                else:
                    shutil.copy(input_path, output_subfolder)
                    # Realizar aumento de datos
                    augment_image(input_path, output_subfolder, num_augmented_images)

# Ruta del conjunto de datos
data_folder = 'ruta entrada imaganes para aplicar data augmentation'
output_folder = 'ruta salida imaganes para aplicar data augmentation'

augment_dataset(data_folder, output_folder)
