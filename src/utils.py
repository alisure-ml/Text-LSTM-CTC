import tensorflow as tf

maxPrintLen = 100

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 180, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('max_stepsize', 8, 'max step size')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_epochs', 1, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', '/home/hp-z840/ALISURE/pycharm/file/Data/image_contest_level_1', 'train')
tf.app.flags.DEFINE_string('infer_dir', '/home/hp-z840/ALISURE/pycharm/file/Data/image_contest_level_1', 'infer')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'infer', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')

FLAGS = tf.app.flags.FLAGS
