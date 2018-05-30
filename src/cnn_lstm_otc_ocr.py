import tensorflow as tf


class LSTMOCR(object):

    def __init__(self, num_classes, batch_size, image_height=60, image_width=180, image_channel=1, out_channels=64,
                 cnn_count=4, num_hidden=128, initial_learning_rate=1e-3, output_keep_prob=0.8,
                 decay_steps=10000, decay_rate=0.98, is_train=True):
        self._is_train = is_train
        self._image_height = image_height
        self._image_width = image_width
        self._num_classes = num_classes
        self._image_channel = image_channel
        self._out_channels = out_channels
        self._cnn_count = cnn_count
        self._batch_size = batch_size
        self._num_hidden = num_hidden
        self._initial_learning_rate = initial_learning_rate
        self._output_keep_prob = output_keep_prob
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

        self.inputs = tf.placeholder(tf.float32, [None, self._image_height, self._image_width, self._image_channel])
        self.labels = tf.sparse_placeholder(tf.int32)
        pass

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()
        pass

    def _build_model(self):
        # CNN
        with tf.variable_scope('cnn'):
            x = self.inputs
            filters = [1, 64, 128, 128, self._out_channels]
            for i in range(self._cnn_count):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides=1)
                    x = self._batch_norm(is_train=self._is_train, name='bn%d' % (i + 1), x=x)
                    x = self._leaky_relu(x)
                    x = self._max_pool(x, 2, strides=2)
                pass
            _, feature_h, feature_w, _ = x.get_shape().as_list()
            print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))
            pass

        # 一维数据，长度为batch_size,值为feature_w
        # 表示每个数据的time_step长度
        self.seq_len = tf.fill([self._batch_size], feature_w)

        # LSTM
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.reshape(x, [self._batch_size, feature_w, feature_h * self._out_channels])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))

            cell = tf.nn.rnn_cell.LSTMCell(self._num_hidden, state_is_tuple=True)
            cell1 = tf.nn.rnn_cell.LSTMCell(self._num_hidden, state_is_tuple=True)
            if self._is_train:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self._output_keep_prob)
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=self._output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(self._batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(cell=stack, inputs=x, sequence_length=self.seq_len,
                                           initial_state=initial_state, dtype=tf.float32, time_major=False)
            pass

        outputs = tf.reshape(outputs, [-1, self._num_hidden])
        w = tf.get_variable('W_out', [self._num_hidden, self._num_classes], tf.float32, tf.glorot_uniform_initializer())
        b = tf.get_variable('b_out', shape=[self._num_classes], dtype=tf.float32, initializer=tf.constant_initializer())

        self.logits = tf.add(tf.matmul(outputs, w), b)
        self.logits = tf.reshape(self.logits, [tf.shape(x)[0], -1, self._num_classes])
        # Time major
        self.logits = tf.transpose(self.logits, (1, 0, 2))
        pass

    def _build_train_op(self, beta1=0.9, beta2=0.999):
        self.global_step = tf.train.get_or_create_global_step()

        self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)

        self.lrn_rate = tf.train.exponential_decay(self._initial_learning_rate, self.global_step,
                                                   self._decay_steps, self._decay_rate, staircase=True)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('learning_rate', self.lrn_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lrn_rate, beta1=beta1,
                                               beta2=beta2).minimize(self.loss, global_step=self.global_step)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
        pass

    @staticmethod
    def _conv2d(x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W', shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='b', shape=[out_channels], dtype=tf.float32, initializer=tf.constant_initializer())
            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')
        return tf.nn.bias_add(con2d_op, b)

    @staticmethod
    def _batch_norm(is_train, name, x):
        with tf.variable_scope(name):
            x_bn = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, center=True, scale=True, epsilon=1e-5,
                                                updates_collections=None, is_training=is_train, fused=True,
                                                data_format='NHWC', zero_debias_moving_mean=True, scope='BatchNorm')
        return x_bn

    @staticmethod
    def _leaky_relu(x, leakiness=0.01):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def _max_pool(x, k_size, strides):
        return tf.nn.max_pool(x, [1, k_size, k_size, 1], [1, strides, strides, 1], padding='SAME', name='max_pool')

    pass
