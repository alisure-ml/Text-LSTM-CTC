import datetime
import os
import time

from PIL import Image
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
from data import DataIterator, NumClasses


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0

    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if origin_label == decoded_label:
            count += 1
        pass

    return count / len(original_seq)


def train(train_dir, batch_size=64, image_height=60, image_width=180, image_channel=1,
          checkpoint_dir="../checkpoint/", num_epochs=100):

    # 加载数据
    train_data = DataIterator(data_dir=train_dir, batch_size=batch_size, begin=0, end=800)
    valid_data = DataIterator(data_dir=train_dir,  batch_size=batch_size, begin=800, end=1000)
    print('train data batch number: {}'.format(train_data.number_batch))
    print('valid data batch number: {}'.format(valid_data.number_batch))

    # 模型
    model = cnn_lstm_otc_ocr.LSTMOCR(NumClasses, batch_size, image_height=image_height,
                                     image_width=image_width, image_channel=image_channel, is_train=True)
    model.build_graph()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(checkpoint_dir + 'train', sess.graph)

        # 加载模型
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from checkpoint{0}'.format(ckpt))
        else:
            print('no checkpoint to restore')
            pass

        print('=======begin training=======')
        for cur_epoch in range(num_epochs):
            start_time = time.time()
            batch_time = time.time()

            # 训练
            train_cost = 0
            for cur_batch in range(train_data.number_batch):
                if cur_batch % 100 == 0:
                    print('batch {}/{} time: {}'.format(cur_batch, train_data.number_batch, time.time() - batch_time))
                batch_time = time.time()

                batch_inputs, _, sparse_labels = train_data.next_train_batch()

                summary, cost, step, _ = sess.run([model.merged_summay, model.cost, model.global_step, model.train_op],
                                                  {model.inputs: batch_inputs, model.labels: sparse_labels})
                train_cost += cost
                train_writer.add_summary(summary, step)
                pass
            print("loss is {}".format(train_cost / train_data.number_batch))

            # 保存模型
            if cur_epoch % 1 == 0:
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                saver.save(sess, os.path.join(checkpoint_dir, 'ocr-model'), global_step=cur_epoch)
                pass

            # 测试
            if cur_epoch % 1 == 0:
                lr = 0
                acc_batch_total = 0
                for j in range(valid_data.number_batch):
                    val_inputs, _, sparse_labels, ori_labels = valid_data.next_test_batch(j)
                    dense_decoded, lr = sess.run([model.dense_decoded, model.lrn_rate],
                                                 {model.inputs: val_inputs, model.labels: sparse_labels})
                    acc_batch_total += accuracy_calculation(ori_labels, dense_decoded, -1)
                    pass

                accuracy = acc_batch_total / valid_data.number_batch

                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, accuracy = {:.3f}, time = {:.3f},lr={:.8f}"
                print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 cur_epoch + 1, num_epochs, accuracy, time.time() - start_time, lr))

            pass
        pass

    pass


def infer(img_path, batch_size=64, image_height=60, image_width=180, image_channel=1, checkpoint_dir="../checkpoint/"):
    # 读取图片的名称
    file_names = os.listdir(img_path)
    file_names = [t for t in file_names if t.find("label") < 0]
    file_names.sort(key=lambda x: int(x.split('.')[0]))
    file_names = np.asarray([os.path.join(img_path, file_name) for file_name in file_names])

    # 模型
    model = cnn_lstm_otc_ocr.LSTMOCR(num_classes=NumClasses, batch_size=batch_size, is_train=False)
    model.build_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # 初始化模型
        sess.run(tf.global_variables_initializer())

        # 加载模型
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            print('restore from ckpt{}'.format(ckpt))
            tf.train.Saver(tf.global_variables(), max_to_keep=100).restore(sess, ckpt)
        else:
            print('cannot restore')
            raise Exception("cannot restore")

        results = []
        for curr_step in range(len(file_names) // batch_size):

            # 读取图片数据
            images_input = []
            for img in file_names[curr_step * batch_size: (curr_step + 1) * batch_size]:
                image_data = np.asarray(Image.open(img).convert("L"), dtype=np.float32) / 255.
                image_data = np.reshape(image_data, [image_height, image_width, image_channel])
                images_input.append(image_data)
            images_input = np.asarray(images_input)

            # 运行得到结果
            # net_results = sess.run(model.dense_decoded, {model.inputs: images_input})
            net_results = sess.run([model.logits, model.seq_len, model.decoded, model.log_prob, model.dense_decoded], {model.inputs: images_input})

            # 对网络输出进行解码得到结果
            for item in net_results:
                result = DataIterator.get_result(item)
                results.append(result)
                print(result)
                pass

            pass

        # 保存结果
        with open('./result.txt', 'a') as f:
            for code in results:
                f.write(code + '\n')
            pass
        pass

    pass


def main(_):
    use_cpu = True
    is_train = False
    # train_dir = "/home/hp-z840/ALISURE/pycharm/file/Data/image_contest_level_1"
    # infer_dir = "/home/hp-z840/ALISURE/pycharm/file/Data/image_contest_level_1"
    train_dir = "C:\\ALISURE\\DataModel\\Data\\ICPR2018\\image_contest_level_1"
    infer_dir = "C:\\ALISURE\\DataModel\\Data\\ICPR2018\\image_contest_level_1"

    dev = '/cpu:0' if use_cpu else '/gpu:0'
    with tf.device(dev):
        if is_train:
            train(train_dir)
        else:
            infer(infer_dir)
        pass

    pass


if __name__ == '__main__':
    tf.app.run()
