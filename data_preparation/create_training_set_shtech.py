import os
import numpy as np
import random
from PIL import Image
import scipy.io as sio
import cv2
from get_density_map_gaussian import get_density_map_gaussian

# Инициализация случайного генератора
seed = 95461354
random.seed(seed)
np.random.seed(seed)

N = 9
dataset = 'B'
dataset_name = f'shanghaitech_part_{dataset}_patches_{N}'
path = f'../data/original/shanghaitech/part_{dataset}_final/train_data/images/'
output_path = '../data/formatted_trainval/'
train_path_img = os.path.join(output_path, dataset_name, 'train/')
train_path_den = os.path.join(output_path, dataset_name, 'train_den/')
val_path_img = os.path.join(output_path, dataset_name, 'val/')
val_path_den = os.path.join(output_path, dataset_name, 'val_den/')
gt_path = f'../data/original/shanghaitech/part_{dataset}_final/train_data/ground_truth/'

# Создание директорий
os.makedirs(output_path, exist_ok=True)
os.makedirs(train_path_img, exist_ok=True)
os.makedirs(train_path_den, exist_ok=True)
os.makedirs(val_path_img, exist_ok=True)
os.makedirs(val_path_den, exist_ok=True)

if dataset == 'A':
    num_images = 300
else:
    num_images = 400

num_val = int(np.ceil(num_images * 0.1))
indices = random.sample(range(1, num_images + 1), num_images)

for idx, i in enumerate(indices, 1):
    if idx % 10 == 0:
        print(f'Processing {idx:3d}/{num_images} files')

    # Загрузка .mat файла с аннотациями
    mat_file_path = os.path.join(gt_path, f'GT_IMG_{i}.mat')
    mat_data = sio.loadmat(mat_file_path)
    image_info = mat_data['image_info']

    # Загрузка изображения
    input_img_name = os.path.join(path, f'IMG_{i}.jpg')
    im = Image.open(input_img_name)
    im = np.array(im)

    # Конвертация в grayscale если нужно
    if len(im.shape) == 3 and im.shape[2] == 3:
        im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])

    h, w = im.shape
    wn2 = w / 8
    hn2 = h / 8
    wn2 = 8 * np.floor(wn2 / 8)
    hn2 = 8 * np.floor(hn2 / 8)

    # Извлечение точек аннотации
    annPoints = image_info[0, 0]['location'][0, 0]

    # Изменение размера изображения если нужно
    if w <= 2 * wn2:
        new_w = int(2 * wn2 + 1)
        im = cv2.resize(im, (new_w, h))
        if len(annPoints) > 0:
            annPoints[:, 0] = annPoints[:, 0] * new_w / w
        w = new_w

    if h <= 2 * hn2:
        new_h = int(2 * hn2 + 1)
        im = cv2.resize(im, (w, new_h))
        if len(annPoints) > 0:
            annPoints[:, 1] = annPoints[:, 1] * new_h / h
        h = new_h

    # Определение области для сэмплирования
    a_w = int(wn2 + 1)
    b_w = int(w - wn2)
    a_h = int(hn2 + 1)
    b_h = int(h - hn2)

    # Создание карты плотности
    im_density = get_density_map_gaussian(im, annPoints)

    for j in range(1, N + 1):
        # Случайные координаты для вырезки
        x = int(np.floor((b_w - a_w) * random.random() + a_w))
        y = int(np.floor((b_h - a_h) * random.random() + a_h))
        x1 = int(x - wn2)
        y1 = int(y - hn2)
        x2 = int(x + wn2 - 1)
        y2 = int(y + hn2 - 1)

        # Вырезка патча из изображения и карты плотности
        im_sampled = im[y1:y2 + 1, x1:x2 + 1]
        im_density_sampled = im_density[y1:y2 + 1, x1:x2 + 1]

        # Фильтрация точек аннотации в вырезке
        if len(annPoints) > 0:
            mask = (annPoints[:, 0] > x1) & (annPoints[:, 0] < x2) & \
                   (annPoints[:, 1] > y1) & (annPoints[:, 1] < y2)
            annPoints_sampled = annPoints[mask].copy()
            annPoints_sampled[:, 0] = annPoints_sampled[:, 0] - x1
            annPoints_sampled[:, 1] = annPoints_sampled[:, 1] - y1
        else:
            annPoints_sampled = np.array([])

        img_idx = f'{i}_{j}'

        # Сохранение в train или val
        if idx <= num_val:
            # Сохранение изображения
            im_sampled_uint8 = (im_sampled * 255).astype(np.uint8) if im_sampled.dtype != np.uint8 else im_sampled
            Image.fromarray(im_sampled_uint8).save(os.path.join(val_path_img, f'{img_idx}.jpg'))
            # Сохранение карты плотности
            np.savetxt(os.path.join(val_path_den, f'{img_idx}.csv'), im_density_sampled, delimiter=',')
        else:
            im_sampled_uint8 = (im_sampled * 255).astype(np.uint8) if im_sampled.dtype != np.uint8 else im_sampled
            Image.fromarray(im_sampled_uint8).save(os.path.join(train_path_img, f'{img_idx}.jpg'))
            np.savetxt(os.path.join(train_path_den, f'{img_idx}.csv'), im_density_sampled, delimiter=',')

print("Обработка завершена!")