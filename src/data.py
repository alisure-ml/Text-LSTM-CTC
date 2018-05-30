import os
from PIL import Image
import numpy as np

Charset = "#0123456789+-*()"
NumClasses = 3 + 2 + 10 + 1 + 1  # +-* + () + 10 digit + blank + space


class DataIterator:

    def __init__(self, data_dir, batch_size, image_height=60, image_width=180, image_channel=1,
                 out_channels=64, begin=0, end=1000):
        self._image_height = image_height
        self._image_width = image_width
        self._image_channel = image_channel
        self._out_channels = out_channels
        self._batch_size = batch_size

        self.image = []
        self.labels = []
        self.labels_result = []

        # 读标签
        with open(os.path.join(data_dir, "labels.txt"), "r") as f:
            labels = f.readlines()
            self.labels_result = [label.split(" ")[1].strip() for label in labels]
            self.labels = [[Charset.find(label_one) for label_one in label.split(" ")[0]] for label in labels]

            # 图片上的内容
            self.labels = self.labels[begin: end]
            # 计算图片上内容后结果
            self.labels_result = self.labels_result[begin: end]
            pass

        for root, _, file_list in os.walk(data_dir):
            # 读取所有的文件
            file_list = [file for file in file_list if file.find(".png") >= 0]
            file_list = sorted(file_list, key=lambda x: int(x.split(".")[0]))
            file_list = file_list[begin: end]

            # 读取数据并处理
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                im = np.asarray(Image.open(image_name).convert("L"), dtype=np.float32) / 255.
                im = np.reshape(im, [image_height, image_width, image_channel])
                self.image.append(im)
            pass

        # 生成随机数
        self.shuffle_idx = np.random.permutation(len(self.labels))
        self.now_batch = 0
        self.number_batch = len(self.labels) // self._batch_size
        pass

    # 根据label id得到结果
    @staticmethod
    def get_result(labels):
        results = ""
        for label in labels:
            if 0 <= label < len(Charset):
                results += Charset[label]
            else:
                results += "$"
                pass
        return results

    def next_train_batch(self):
        # 一批数据结束了
        if self.now_batch >= self.number_batch:
            self.shuffle_idx = np.random.permutation(len(self.labels))
            self.now_batch = 0
            pass

        # 选择数据
        index = [self.shuffle_idx[i % len(self.labels)] for i in range(self.now_batch * self._batch_size,
                                                                       (self.now_batch + 1) * self._batch_size)]
        self.now_batch += 1

        # 图片数据
        image_batch = [self.image[i] for i in index]
        image_batch = np.array(image_batch)
        # 图片标签
        sparse_labels = self.sparse_tuple_from_label([self.labels[i] for i in index])
        # 特征图长度
        feature_lengths = np.asarray([self._out_channels for _ in image_batch], dtype=np.int64)
        return image_batch, feature_lengths, sparse_labels

    def next_test_batch(self, index):
        index = 0 if index >= self.number_batch else index
        # 图片数据
        image_batch = self.image[index * self._batch_size: (index + 1) * self._batch_size]
        image_batch = np.array(image_batch)
        # 特征图长度
        feature_lengths = np.asarray([self._out_channels for _ in image_batch], dtype=np.int64)
        # 图片标签
        label_batch = self.labels[index * self._batch_size: (index + 1) * self._batch_size]
        sparse_labels = self.sparse_tuple_from_label(label_batch)
        return image_batch, feature_lengths, sparse_labels, label_batch

    # 转换成ctc_loss所需的格式
    @staticmethod
    def sparse_tuple_from_label(batch_labels, dtype=np.int32):
        indices = []
        values = []

        for n, seq in enumerate(batch_labels):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(batch_labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    pass
