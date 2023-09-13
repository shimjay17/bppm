import logging
import cv2
import numpy as np
import os
import ipdb

def truncate_image_vertical(img, cut_upper, cut_lower):
    H = img.shape[0]
    
    start_idx = cut_upper
    end_idx = H - cut_lower

    img_truncated = img[start_idx:end_idx]
    return img_truncated

def get_mask(src_img_path, color_thres=35, apply_opening=True, opening_kernel_size=15):
    image = cv2.imread(src_img_path)
    mask = (image[:, :, 0] <= color_thres) & (image[:, :, 1] <= color_thres) & (image[:, :, 2] <= color_thres).astype(np.uint8)
    mask = (mask * 255).astype('uint8')
    if apply_opening:
        kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = 255 - mask
    return mask

def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    high_pass = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    mask = np.abs(image - high_pass) >= threshold
    sharpened = np.where(mask, high_pass, image)
    return sharpened.astype(np.uint8)

def crop_and_save(img_dir, mask, save_dir, crop_frac):
    img = cv2.imread(img_dir)
    #cv2.imwrite(os.path.join(os.path.dirname(save_dir), 'mask_' + os.path.basename(save_dir)), mask)
    mask[mask==255] = 1

    row_fracs = mask.mean(axis=1)
    col_fracs = mask.mean(axis=0)

    rows_to_keep = row_fracs > crop_frac
    cols_to_keep = col_fracs > crop_frac

    true_indices = np.where(rows_to_keep)[0]
    if true_indices.size:
        start, end = true_indices[0], true_indices[-1]
        rows_to_keep[start:end+1] = True 
    true_indices = np.where(cols_to_keep)[0]
    if true_indices.size:
        start, end = true_indices[0], true_indices[-1]
        cols_to_keep[start:end+1] = True 

    cropped_img = img[rows_to_keep][:, cols_to_keep]
    cv2.imwrite(save_dir, cropped_img)
    return cropped_img

def resize_save_image(image_path, rs_save_path, output_shape):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, output_shape)
    cv2.imwrite(rs_save_path, resized_img)
    return None
