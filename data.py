import glob
import os
import cv2
import re
import time
import numpy as np
import tensorflow as tf
from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string("train_data_path", "data/train_img/", "training dataset")
tf.app.flags.DEFINE_string("train_gt_path", "data/train_gt/", "training dataset")


FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ["jpg", "png", "bmp"]:
        files.extend(glob.glob(os.path.join(FLAGS.train_data_path, "*.{}".format(ext))))
    return files


def crop(image, width=256, heigh=256):
    return image[:, :, :]


def generator(batch_size=8):
    image_list = get_images()
    while True:
        images = []
        gt_image = []
        try:
            for image_name in image_list:
                image = cv2.imread(image_name)
                # image = crop(image)
                # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gt_name = re.sub("image", "mask", image_name)
                # image_gt_name = re.sub(re.split("\.", image_name)[1], "png", image_gt_name)
                image_gt_name = FLAGS.train_gt_path+image_gt_name.split("\\")[-1]
                # if image_gt_name.split("\\")[-1] not in os.listdir(FLAGS.train_gt_path):
                #     continue
                image_gt = cv2.imread(image_gt_name)
                image_gt_gray = cv2.cvtColor(image_gt, cv2.COLOR_BGR2GRAY)
                image_gt_gray = np.where(image_gt_gray > 100, 255.0, 0.0)
                image_gt_gray = image_gt_gray.reshape((image_gt_gray.shape[0], image_gt_gray.shape[1], 1))
                # image_gt_gray = crop(image_gt_gray)
                images.append(image/255.0)
                gt_image.append(image_gt_gray/255.0)
                if len(images) == batch_size:
                    yield images, gt_image
                    images = []
                    gt_image = []
        except Exception as e:
            continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=False)
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    pass
