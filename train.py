import os
import tensorflow as tf
import unet
import data

tf.app.flags.DEFINE_string("checkpoint_path", "ckpt/membrane/", "store the model.")
tf.app.flags.DEFINE_float("learning_rate", "0.0001", "")
tf.app.flags.DEFINE_integer("batch_size", 4, "")
tf.app.flags.DEFINE_integer("number_reasers", 1, "")
tf.app.flags.DEFINE_integer("max_step", 100, "")

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    gt_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="input_gt")
    unet_output = unet.unet(name="UNET", input_data=input_images)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(gt_maps, unet_output))
    train_ops = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())

    summaty_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    init = tf.global_variables_initializer()

    with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(init)
        data_generator = data.get_batch(num_workers=FLAGS.number_reasers,
                                        batch_size=FLAGS.batch_size)
        for step in range(FLAGS.max_step):
            input_list = next(data_generator)
            peer_loss, _ = sess.run([loss, train_ops],
                                    feed_dict={input_images: input_list[0],gt_maps: input_list[1]})
            print("step {}, model loss {}".format(step, peer_loss))
            saver.save(sess=sess, save_path=FLAGS.checkpoint_path + str(step) + ".ckpt", global_step=step)


if __name__ == "__main__":
    tf.app.run()
