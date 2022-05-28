import numpy as np
import os
import cv2
from tqdm import tqdm
import time
from multiprocessing import Pool


def img_sub(img_path, sub_path, crop_size, step):
    img_name = os.path.basename(img_path)
    # 图像编号与后缀名
    img_name, suffix = img_name.split(".")
    img_name = img_name.replace("x2", "").replace("x3", "").replace("x4", "")
    # 读入原图像
    img = cv2.imread(img_path, -1)
    h, w, c = img.shape

    h_sub_point = np.arange(0, h - crop_size + 1, step)
    w_sub_point = np.arange(0, w - crop_size + 1, step)

    # 若最后分割起始点小于h-crop_size的长度，则加入位于h-crop_size处的分割起始点
    if h - (h_sub_point[-1] + crop_size) > 0:
        h_sub_point = np.append(h_sub_point, h - crop_size)
    if w - (w_sub_point[-1] + crop_size) > 0:
        w_sub_point = np.append(w_sub_point, w - crop_size)

    index = 0
    for x in h_sub_point:
        for y in w_sub_point:
            index += 1
            crop_img = img[x:x + crop_size, y:y + crop_size, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(os.path.join(sub_path, "{}_{:03d}.{}".format(img_name, index, suffix)), crop_img
                        , [cv2.IMWRITE_PNG_COMPRESSION, 3])

    # process_info = 'Processing {} ...'.format(img_name)
    # print(process_info)


def main(input_path, sub_path, crop_size, step):
    start_time = time.time()

    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
        print("Folder is being created:{}".format(sub_path))
    else:
        print("Folder {} already exists.".format(sub_path))

    img_name = os.listdir(input_path)
    img_path_list = []
    for i, name in enumerate(img_name):
        str1 = os.path.join(input_path, name)
        img_path_list.append(str1)

    pbar = tqdm(total=len(img_path_list), unit="image", desc="extract")

    # 进程数量
    pool = Pool(20)
    for path in img_path_list:
        # 多进程异步分割图片
        pool.apply_async(img_sub, args=(path, sub_path, crop_size, step), callback=lambda arg: pbar.update(1))
        # img_sub(path, sub_path, crop_size, step)

    pool.close()
    pool.join()
    pbar.close()
    # 多进程处理时间仅为不使用多进程处理时间的1/4
    print('All processes done! The total time is:{:.2f}s'.format(time.time() - start_time))


if __name__ == "__main__":
    # HR
    input_path = "../../data/DIV2K_HR/DIV2K_train_HR"
    sub_path = "../../data/DIV2K_HR/DIV2K_train_HR_sub"
    crop_size = 480
    step = 240
    main(input_path, sub_path, crop_size, step)

    # LR_x2
    input_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X2"
    sub_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X2_sub"
    crop_size = 240
    step = 120
    main(input_path, sub_path, crop_size, step)

    # LR_x3
    input_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X3"
    sub_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X3_sub"
    crop_size = 160
    step = 80
    main(input_path, sub_path, crop_size, step)

    # LR_x4
    input_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X4"
    sub_path = "../../data/DIV2K_HR/DIV2K_train_LR_bicubic/X4_sub"
    crop_size = 120
    step = 60
    main(input_path, sub_path, crop_size, step)
