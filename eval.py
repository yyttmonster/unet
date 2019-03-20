import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import unet

tf.app.flags.DEFINE_string("ckpt_restore_path", "ckpt/membrane/", "")
tf.app.flags.DEFINE_string("test_data_path", "data/eval/", "")
tf.app.flags.DEFINE_string("output_path", "data/eval_output/", "")

FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ["jpg", "png", "bmp"]:
        files.extend(glob.glob(os.path.join(FLAGS.test_data_path, "*.{}".format(ext))))
    return files


def main(argv=None):
    unet_graph = tf.Graph()
    with unet_graph.as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
        unet_output = unet.unet(name="UNET", input_data=input_images)
        saver = tf.train.Saver(tf.global_variables())

    with tf.Session(graph=unet_graph) as sess:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.ckpt_restore_path)
        model_path = os.path.join(FLAGS.ckpt_restore_path, os.path.basename(ckpt_state.model_checkpoint_path))
        saver.restore(sess=sess, save_path=model_path)

        image_list = get_images()
        for img_name in image_list:
            im = cv2.imread(img_name)[:,:,::-1]
            output = sess.run(unet_output,feed_dict={input_images:[im/255.0]})
            coordinates = np.argmax(output[0],axis=-1)
            image = np.where(coordinates!=0,255,0)
            print(img_name,coordinates)
            cv2.imwrite(FLAGS.output_path+"a.jpg",image)
            # cv2.imshow("a",image)
            # cv2.waitKey()



if __name__ == "__main__":
    tf.app.run()