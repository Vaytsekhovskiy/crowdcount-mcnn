import os
import numpy as np
from PIL import Image
import scipy.io as sio
from get_density_map_gaussian import get_density_map_gaussian

# Параметры датасета
dataset = 'B'
dataset_name = f'shanghaitech_part_{dataset}'
path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/images/'
gt_path = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth/'
gt_path_csv = f'../data/original/shanghaitech/part_{dataset}_final/test_data/ground_truth_csv/'

# Создаем директорию если не существует
os.makedirs(gt_path_csv, exist_ok=True)

if dataset == 'A':
    num_images = 182
else:
    num_images = 316

# Основной цикл обработки изображений
for i in range(1, num_images + 1):
    if i % 10 == 0:
        print(f'Processing {i:3d}/{num_images} files')

    # Загружаем MAT файл с аннотациями
    mat_file_path = os.path.join(gt_path, f'GT_IMG_{i}.mat')
    mat_data = sio.loadmat(mat_file_path)
    image_info = mat_data['image_info']

    # Загружаем изображение
    input_img_name = os.path.join(path, f'IMG_{i}.jpg')
    im = Image.open(input_img_name)
    im = np.array(im)

    # Конвертируем в grayscale если нужно
    if len(im.shape) == 3 and im.shape[2] == 3:
        im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale

    # Извлекаем точки аннотации
    annPoints = image_info[0, 0]['location'][0, 0]

    # Получаем карту плотности
    im_density = get_density_map_gaussian(im, annPoints)

    # Сохраняем в CSV
    csv_file_path = os.path.join(gt_path_csv, f'IMG_{i}.csv')
    np.savetxt(csv_file_path, im_density, delimiter=',')