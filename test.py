import tensorflow as tf
from train import get_data,features_from_data

flags = tf.app.flags
flags.DEFINE_string("train_models","./train_models","path to save checkpoints")
FLAGS = flags.FLAGS


def test():
    data = get_data(is_test=True)
    x, test_labels = features_from_data(data,is_test=True)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_models)
        saver.restore(sess, ckpt.model_checkpoint_path)
        accuracy = tf.get_variable(name="accuracy")
        print("test accuracy %g" % (accuracy.eval(feed_dict={x_:x, y_: test_labels, keep_prob: 1.0})))


def main(argv=None):
    #if not tf.gfile.Exists(FLAGS.test_models):
    #    tf.gfile.makeDirs(FLAGS.test_models)
    test()


if __name__ == "__main__":
    tf.app.run()