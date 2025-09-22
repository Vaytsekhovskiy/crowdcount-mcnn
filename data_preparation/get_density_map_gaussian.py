import numpy as np
import cv2


def get_density_map_gaussian(im, points):
    """
    Создает карту плотности с гауссовыми ядрами
    Args:
        im: входное изображение (только для получения размеров)
        points: массив точек [[x1, y1], [x2, y2], ...]
    Returns:
        im_density: карта плотности
    """
    h, w = im.shape[:2]
    im_density = np.zeros((h, w))

    if len(points) == 0:
        return im_density

    if len(points) == 1:
        x1 = max(0, min(w - 1, int(round(points[0, 0]))))
        y1 = max(0, min(h - 1, int(round(points[0, 1]))))
        im_density[y1, x1] = 255
        return im_density

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0

        # Создаем гауссово ядро
        H = cv2.getGaussianKernel(f_sz, sigma)
        H = H @ H.T  # делаем 2D ядро

        x = min(w - 1, max(0, int(np.floor(points[j, 0]))))
        y = min(h - 1, max(0, int(np.floor(points[j, 1]))))

        if x >= w or y >= h:
            continue

        # Определяем координаты для размещения гауссова ядра
        x1 = x - f_sz // 2
        y1 = y - f_sz // 2
        x2 = x1 + f_sz
        y2 = y1 + f_sz

        # Корректируем границы
        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change_H = False

        if x1 < 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        if y1 < 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True

        # Если нужно изменить размер ядра из-за границ
        if change_H:
            new_h = y2 - y1
            new_w = x2 - x1
            if new_h > 0 and new_w > 0:
                H = cv2.getGaussianKernel(new_w, sigma)
                H = H @ H.T
                if new_h != new_w:
                    H = cv2.resize(H, (new_w, new_h))
            else:
                continue

        # Добавляем ядро к карте плотности
        im_density[y1:y2, x1:x2] += H

    return im_density