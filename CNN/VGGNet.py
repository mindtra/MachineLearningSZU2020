import tensorflow as tf
import utils
from time import time
import os
import numpy as np


def train(file_dir, retrain):
    with tf.device("/gpu:1"):
        startTime = time()

        x = tf.placeholder(shape=[None, 398, 796, channels], dtype=tf.float32)
        y = tf.placeholder(shape=[None, class_nums], dtype=tf.int32)

        with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
            w1 = tf.Variable(tf.random_normal([3, 3, 1, 64]), dtype=tf.float32, name="w")
            c1 = tf.nn.conv2d(x, w1, [1, 1, 1, 1], 'SAME')
            b1 = tf.Variable(tf.random_normal([64]), dtype=tf.float32, name='b')
            c1 = tf.nn.bias_add(c1, b1)
            c1 = tf.nn.relu(c1)
        with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
            w2 = tf.Variable(tf.random_normal([3, 3, 64, 64]), dtype=tf.float32, name='w')
            c2 = tf.nn.conv2d(c1, w2, [1, 1, 1, 1], padding="SAME")
            b2 = tf.Variable(tf.random_normal([64]), dtype=tf.float32, name='b')
            c2 = tf.nn.bias_add(c2, b2)
            c2 = tf.nn.relu(c2)
        p_c2 = tf.nn.max_pool(c2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        # p_c2 199*299*64
        with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
            w3 = tf.Variable(tf.random_normal([3, 3, 64, 128]), dtype=tf.float32, name='w')
            c3 = tf.nn.conv2d(p_c2, w3, [1, 1, 1, 1], "SAME")
            b3 = tf.Variable(tf.random_normal([128]), dtype=tf.float32, name='b')
            c3 = tf.nn.bias_add(c3, b3)
            c3 = tf.nn.relu(c3)
        with tf.variable_scope("layer4", reuse=tf.AUTO_REUSE):
            w4 = tf.Variable(tf.random_normal([3, 3, 128, 128]), dtype=tf.float32, name='w')
            c4 = tf.nn.conv2d(c3, w4, [1, 1, 1, 1], "SAME")
            b4 = tf.Variable(tf.random_normal([128]), dtype=tf.float32, name='b')
            c4 = tf.nn.bias_add(c4, b4)
            c4 = tf.nn.relu(c4)
        p_c4 = tf.nn.max_pool(c4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        # p_c4 100*150*128
        with tf.variable_scope("layer5", reuse=tf.AUTO_REUSE):
            w5 = tf.Variable(tf.random_normal([3, 3, 128, 256]), dtype=tf.float32, name='w')
            c5 = tf.nn.conv2d(p_c4, w5, [1, 1, 1, 1], "SAME")
            b5 = tf.Variable(tf.random_normal([256]), dtype=tf.float32, name="b")
            c5 = tf.nn.bias_add(c5, b5)
            c5 = tf.nn.relu(c5)
        with tf.variable_scope("layer6", reuse=tf.AUTO_REUSE):
            w6 = tf.Variable(tf.random_normal([3, 3, 256, 256]), dtype=tf.float32, name="w")
            c6 = tf.nn.conv2d(c5, w6, [1, 1, 1, 1], "SAME")
            b6 = tf.Variable(tf.random_normal([256]), dtype=tf.float32, name="b")
            c6 = tf.nn.bias_add(c6, b6)
            c6 = tf.nn.relu(c6)
            # c6 100*150*256
        with tf.variable_scope("layer7", reuse=tf.AUTO_REUSE):
            w7 = tf.Variable(tf.random_normal([3, 3, 256, 256]), dtype=tf.float32, name="w")
            c7 = tf.nn.conv2d(c6, w7, [1, 1, 1, 1], "SAME")
            b7 = tf.Variable(tf.random_normal([256]), dtype=tf.float32, name="b")
            c7 = tf.nn.bias_add(c7, b7)
            c7 = tf.nn.relu(c7)
        p_c7 = tf.nn.max_pool(c7, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # p_c7 50*75*256
        with tf.variable_scope("layer8", reuse=tf.AUTO_REUSE):
            w8 = tf.Variable(tf.random_normal([3, 3, 256, 512]), dtype=tf.float32, name="w")
            c8 = tf.nn.conv2d(p_c7, w8, [1, 1, 1, 1], "SAME")
            b8 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c8 = tf.nn.bias_add(c8, b8)
            c8 = tf.nn.relu(c8)
        with tf.variable_scope("layer9", reuse=tf.AUTO_REUSE):
            w9 = tf.Variable(tf.random_normal([3, 3, 512, 512]), dtype=tf.float32, name="w")
            c9 = tf.nn.conv2d(c8, w9, [1, 1, 1, 1], "SAME")
            b9 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c9 = tf.nn.bias_add(c9, b9)
            c9 = tf.nn.relu(c9)
        with tf.variable_scope("layer10", reuse=tf.AUTO_REUSE):
            w10 = tf.Variable(tf.random_normal([3, 3, 512, 512]), dtype=tf.float32, name="w")
            c10 = tf.nn.conv2d(c9, w10, [1, 1, 1, 1], "SAME")
            b10 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c10 = tf.nn.bias_add(c10, b10)
            c10 = tf.nn.relu(c10)
        p_c10 = tf.nn.max_pool(c10, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # p_c10 25*38*512
        with tf.variable_scope("layer11", reuse=tf.AUTO_REUSE):
            w11 = tf.Variable(tf.random_normal([3, 3, 512, 512]), dtype=tf.float32, name="w")
            c11 = tf.nn.conv2d(p_c10, w11, [1, 1, 1, 1], "SAME")
            b11 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c11 = tf.nn.bias_add(c11, b11)
            c11 = tf.nn.relu(c11)
        with tf.variable_scope("layer12", reuse=tf.AUTO_REUSE):
            w12 = tf.Variable(tf.random_normal([3, 3, 512, 512]), dtype=tf.float32, name="w")
            c12 = tf.nn.conv2d(c11, w12, [1, 1, 1, 1], "SAME")
            b12 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c12 = tf.nn.bias_add(c12, b12)
            c12 = tf.nn.relu(c12)
        with tf.variable_scope("layer13", reuse=tf.AUTO_REUSE):
            w13 = tf.Variable(tf.random_normal([3, 3, 512, 512]), dtype=tf.float32, name="w")
            c13 = tf.nn.conv2d(c12, w13, [1, 1, 1, 1], "SAME")
            b13 = tf.Variable(tf.random_normal([512]), dtype=tf.float32, name="b")
            c13 = tf.nn.bias_add(c13, b13)
            c13 = tf.nn.relu(c13)
        p_c13 = tf.nn.max_pool(c13, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # p_c10 13*19*512
        shape = p_c13.get_shape()
        flatten_p_13 = tf.reshape(p_c13, [-1, shape[1].value * shape[2].value * shape[3].value])

        # BN
        epsilon = 10e-5
        mean, var = tf.nn.moments(flatten_p_13, axes=[0])
        flatten_p_13_bn = tf.nn.batch_normalization(x=flatten_p_13, mean=mean, variance=var, offset=0, scale=1,
                                                    variance_epsilon=epsilon)

        with tf.variable_scope("layer14", reuse=tf.AUTO_REUSE):
            w14 = tf.Variable(tf.random_normal([shape[1].value * shape[2].value * shape[3].value, 4096]),
                              dtype=tf.float32, name="w")
            fc14 = tf.matmul(flatten_p_13_bn, w14)
            b14 = tf.Variable(tf.random_normal([4096]), dtype=tf.float32, name="b")
            fc14 = tf.nn.bias_add(fc14, b14)
            fc14 = tf.nn.relu(fc14)
        with tf.variable_scope("layer15", reuse=tf.AUTO_REUSE):
            w15 = tf.Variable(tf.random_normal([4096, 256]), dtype=tf.float32, name="w")
            fc15 = tf.matmul(fc14, w15)
            b15 = tf.Variable(tf.random_normal([256]), dtype=tf.float32, name="b")
            fc15 = tf.nn.bias_add(fc15, b15)
            fc15 = tf.nn.relu(fc15)
        with tf.variable_scope("layer16", reuse=tf.AUTO_REUSE):
            w16 = tf.Variable(tf.random_normal([256, class_nums]), dtype=tf.float32, name="w")
            fc16 = tf.matmul(fc15, w16)
            b16 = tf.Variable(tf.random_normal([class_nums]), dtype=tf.float32, name="b")
            fc16 = tf.nn.bias_add(fc16, b16)
            fc16 = tf.nn.relu(fc16)
        # pre = tf.nn.softmax(fc16)
        # define the loss function
        loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc16 + 10e-8, labels=y))
        # loss_function = -tf.reduce_mean(y * tf.log(tf.clip_by_value(pre, 1e-8, 1.0)))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss_function)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        # config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        # config.gpu_options.allow_growth = True

        # with tf.Session(config=config) as sess:   #with tf.Session(config=config) as sess
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        if retrain == True:
            saver.restore(sess, "/home/l/model/Step_000050.ckpt")  # 加载模型
        else:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        epoch_start_time = time()

        flag = 0
        xs, ys = utils.get_file(file_dir)
        for i in range(2000):
            if flag == 0:
                xs, ys = utils.get_file(file_dir)

            x_batch, y_batch = utils.get_batch(xs, ys, batch_size, flag)

            if flag < len(ys) - batch_size:
                flag += batch_size
            else:
                flag = 0

            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
            # 计算准确率
            pre = tf.nn.softmax(fc16)
            correct_prediction = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

            acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})
            print("Now the accuracy is %f" % acc)

            # 计算loss function
            loss = sess.run(loss_function, feed_dict={x: x_batch, y: y_batch})
            print("Now the loss is %f" % loss)

            if flag == 0:
                epoch_end_time = time()
                print("Current epoch takes: ", (epoch_end_time - epoch_start_time))
                epoch_start_time = epoch_end_time

            if (i + 1) % 25 == 0:
                saver.save(sess, os.path.join("./model/", 'Step_{:06d}.ckpt'.format(i + 1)))
            print("------Training step %d is finished------" % (i + 1))

        saver.save(sess, './model/')
        duration = time() - startTime
        print("Train Finished takes:", "{:.2f}".format(duration))

        coord.request_stop()
        coord.join(threads)

        sess.close()


if __name__ == '__main__':
    class_nums = 11
    channels = 1
    batch_size = 32
    capacity = 256
    file_dir = "./data/train/"
    train(file_dir, False)
