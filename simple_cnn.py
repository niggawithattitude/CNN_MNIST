# 简单卷积神经网络对MNIST数据集进行分类: conv2d + activation + pool +fc
import csv
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置算法超参数
learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

# Network Parameters
n_input = 784
n_classes = 10


# 根据指定的维数返回初始化好的指定名称的权重 Variable
def WeightsVariable(shape, name_str, stddev=0.1):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


# 根据指定的维数返回初始化好的指定名称的偏置 Variable
def BiasesVariable(shape, name_str, stddev=0.00001):
    initial = tf.random_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


# 2维卷积层(conv2d-bias)的封装
def Conv2d(x, W, b, stride=1, padding='SAME'):
    with tf.name_scope('Wx_b'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)
    return y


# 非线性激活层的封装
def Activation(x, activation=tf.nn.relu, name='relu'):
    with tf.name_scope(name):
        y = activation(x)
    return y


# 2维池化层pool的封装
def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')


# 全连接层activate(wx+b)的封装
def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)
    with tf.name_scope(act_name):
        y = activate(y)
    return y


#
def EvaluateModelOnDataset(sess, images, labels):
    n_samples = images.shape[0]
    per_batch_size = batch_size
    loss = 0
    acc = 0
    # 样本量比较少的时候，一次性评估完毕；否则拆成若干个批次评估，防止内存不够用
    if(n_samples <= per_batch_size):
        batch_count = 1
        loss, acc = sess.run([cross_entropy_loss, accuracy], feed_dict={X_origin: images,
                                                                        Y_true: labels,
                                                                        learning_rate: learning_rate_init})
    else:
        batch_count = int(n_samples / per_batch_size)
        batch_start = 0
        for idx in range(batch_count):
            batch_loss, batch_acc = sess.run([cross_entropy_loss, accuracy],
                                             feed_dict={X_origin: images[batch_start: batch_start + per_batch_size, :],
                                                        Y_true: labels[batch_start:batch_start + per_batch_size, :],
                                                        learning_rate: learning_rate_init})
            batch_start += per_batch_size
            # 累计所有批次上的损失和准确率
            loss += batch_loss
            acc += batch_acc
    # 返回平均值
    return loss / batch_count, acc / batch_count


with tf.Graph().as_default():
    # 计算图输入
    with tf.name_scope('Input'):
        X_origin = tf.placeholder(tf.float32, [None, n_input], name='X_origin')
        Y_true = tf.placeholder(tf.float32, [None, n_classes], name='Y_true')
        # 把图像数据从N*784的张量转换为N*28*28*1的张量
        X_image = tf.reshape(X_origin, [-1, 28, 28, 1])
    # 计算图前向推断过程
    with tf.name_scope('Inference'):
        # 第一个卷积层(con2d - biase)
        with tf.name_scope('Conv2d'):
            # 5*5  通道数为1  16为滤波器数量
            conv1_kernels_num = 5
            weights = WeightsVariable(shape=[5, 5, 1, conv1_kernels_num], name_str='weights')
            biases = BiasesVariable(shape=[conv1_kernels_num], name_str='biases')
            conv_out = Conv2d(X_image, weights, biases, stride=1, padding='VALID')
            # conv_out为24*24*16
        # 非线性激活层
        with tf.name_scope('Activate'):
            activate_out = Activation(conv_out, activation=tf.nn.relu, name='relu')
        # 第一个池化层(' max Pool 2d')
        with tf.name_scope('Pool2d'):
            # 2*2
            pool_out = Pool2d(activate_out, pool=tf.nn.max_pool, k=2, stride=2)
        # 将二维特征图变换为一维特征向量
        with tf.name_scope('FeatsReshape'):
            # 池化完输出 12*12*16
            features = tf.reshape(pool_out, [-1, 12 * 12 * conv1_kernels_num])
        with tf.name_scope('FC_Linear'):
            weights = WeightsVariable(shape=[12 * 12 * conv1_kernels_num, n_classes], name_str='weights')
            biases = BiasesVariable(shape=[n_classes], name_str='biases')
            Ypred_logits = FullyConnected(features, weights, biases, activate=tf.identity, act_name='identity')

    # 定义损失层
    with tf.name_scope('Loss'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                             (labels=Y_true, logits=Ypred_logits))

    # 定义优化训练层
    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainer = optimizer.minimize(cross_entropy_loss)

    # 定义模型评估层
    with tf.name_scope('Evaluate'):
        correct_pred = tf.equal(tf.argmax(Ypred_logits, 1), tf.argmax(Y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 添加所有变量的初始化节点
    init = tf.global_variables_initializer()

    print('把计算图写入事件文件，在tensorboard里查看')
    summary_writer = tf.summary.FileWriter(logdir='logs/simple_cnn/', graph=tf.get_default_graph())
    summary_writer.close()

    # 导入 MNIST_data
    mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
    # 将评估结果保存到文件
    result_list = list()
    # 写入参数配置
    result_list.append(['learning_rate', learning_rate_init,
                        'training_epochs', training_epochs,
                        'batch_size', batch_size,
                        'display_step', display_step])

    result_list.append(['train_step', 'train_loss', 'validation_loss',
                        'train_step', 'train_accuracy', 'validation_accuracy'])

    # 启动计算图
    with tf.Session().as_default() as sess:
        sess.run(init)
        # 55000 / 100
        total_batches = int(mnist.train.num_examples / batch_size)
        print("Per batch Size: ", batch_size)
        print("Train sample Count: ", mnist.train.num_examples)
        print("Total batch Count: ", total_batches)
        training_step = 0     # 记录模型被训练的步数
        # 训练指定轮数，每一轮所有训练样本都要过一遍
        for epoch in range(training_epochs):
            # 每一轮都要把所有batch跑一遍
            for batch_idx in range(total_batches):
                # 取出数据
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # 运行优化器训练节点(backprop)
                sess.run(trainer, feed_dict={X_origin: batch_x,
                                             Y_true: batch_y,
                                             learning_rate: learning_rate_init})
                # 每调用一次训练节点，training_step+1， 最终==training_epochs*total_batch
                training_step += 1

                # 每训练display_step次，计算当前模型的损失和分类准确率
                if training_step % display_step == 0:
                    # 计算当前模型在目前（最近）见过的display_step个batchsize的训练集上的损失和分类准确率
                    start_idx = max(0, (batch_idx-display_step)*batch_size)
                    end_idx = batch_idx*batch_size
                    train_loss, train_acc = EvaluateModelOnDataset(sess,
                                                                   mnist.train.images[start_idx:end_idx, :],
                                                                   mnist.train.labels[start_idx:end_idx, :])
                    print("Training Step: " + str(training_step) +
                          ", Training Loss=" + "{:.6f}".format(train_loss) +
                          ", Training Accuracy=" + "{:.5f}".format(train_acc))

                    # 计算当前模型在验证集的损失和分类准确率
                    validation_loss, validation_acc = EvaluateModelOnDataset(sess,
                                                                   mnist.validation.images,
                                                                   mnist.validation.labels)
                    print("Training Step: " + str(training_step) +
                          ", Validation Loss=" + "{:.6f}".format(validation_loss) +
                          ", Validation Accuracy=" + "{:.5f}".format(validation_acc))
                    # 将评估结果保存到文件
                    result_list.append([training_step, train_loss, validation_loss,
                                        training_step, train_acc, validation_acc])

    print("训练完毕")

    # 计算指定数量的测试集上的准确率  1000个图像
    test_samples_count = mnist.test.num_examples
    test_loss, test_accuracy = EvaluateModelOnDataset(sess, mnist.test.images, mnist.test.labels)
    print("Testing Samples Count", test_samples_count)
    print("Testing Loss:", test_loss)
    print("Testing Accuracy:", test_accuracy)
    result_list.append(['test step', 'loss', test_loss, 'accuracy', test_accuracy])
    sess.close()
    print("会话关闭")

    # 将评估结果保存到文件
    results_file = open('evaluate_results.cvs', 'w', newline='')
    csv_writer = csv.writer(results_file, dialect='excel')
    for row in result_list:
        csv_writer.writerow(row)

