import tensorflow as tf


def unet(name, input_data):
    # with tf.variable_scope(name, reuse=reuse):
    # layer 1
    # the kernel_initializer is to promise the same size of gradient in every layer
    with tf.variable_scope('layer1'):
        with tf.variable_scope('convolution'):
            conv1_1 = tf.layers.conv2d(inputs=input_data, filters=64, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv1_1')
            conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv1_2')
        # crop1 = tf.keras.layers.Cropping2D(cropping=((24, 24), (24, 24)), name='crop1')(conv1_2)
        pooling1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=[2, 2], name='pooling1')
        print('pooling1:', pooling1.shape)

    # layer 2
    with tf.variable_scope('layer2'):
        with tf.variable_scope('convolution'):
            conv2_1 = tf.layers.conv2d(inputs=pooling1, filters=128, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv2_1')
            conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv1_2')
        pooling2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=[2, 2], name='pooling2')
        print('pooling2:', pooling2.shape)

    # layer 3
    with tf.variable_scope('layer3'):
        with tf.variable_scope('convolution'):
            conv3_1 = tf.layers.conv2d(inputs=pooling2, filters=256, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv3_1')
            conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv3_2')
            dropout3 = tf.layers.dropout(conv3_2)
        pooling3 = tf.layers.max_pooling2d(inputs=dropout3, pool_size=[2, 2], strides=[2, 2], name='pooling3')
        print('pooling3:', pooling3.shape)

        # layer 4
    with tf.variable_scope('layer4'):
        with tf.variable_scope('convolution'):
            conv4_1 = tf.layers.conv2d(inputs=pooling3, filters=512, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv4_1')
            conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='conv4_2')
            dropout4 = tf.layers.dropout(conv4_2)
        pooling4 = tf.layers.max_pooling2d(inputs=dropout4, pool_size=[2, 2], strides=[2, 2], name='pooling4')
        print('pooling4:', pooling4.shape)

    with tf.variable_scope('layer5'):
        conv5_1 = tf.layers.conv2d(inputs=pooling4, filters=1024, kernel_size=3, strides=(1, 1),
                                   activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   name='conv5_1')
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=1024, kernel_size=3, strides=(1, 1),
                                   activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   name='conv5_2')
        print("conv5_2:",conv5_2.shape)

    # upsample 1
    with tf.variable_scope('upsample1'):
        upsample1_1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample')(conv5_2)
        upsample1 = tf.layers.conv2d(upsample1_1, 512, 2, padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name='up_conv1')
        mergeu1_c4 = tf.concat(values=[dropout4, upsample1], axis=3, name='mergeu1_c4')
        up_conv1_1 = tf.layers.conv2d(inputs=mergeu1_c4, filters=512, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv1_1')
        up_conv1_2 = tf.layers.conv2d(inputs=up_conv1_1, filters=512, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv1_2')
    # upsample 2
    with tf.variable_scope('upsample2'):
        upsample2_1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample')(up_conv1_2)
        upsample2 = tf.layers.conv2d(upsample2_1, 128, 2, padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name='up_conv2')
        mergeu2_c3 = tf.concat(values=[dropout3, upsample2], axis=3, name='mergeu2_c3')
        up_conv2_1 = tf.layers.conv2d(inputs=mergeu2_c3, filters=128, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv2_1')
        up_conv2_2 = tf.layers.conv2d(inputs=up_conv2_1, filters=128, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv2_2')

    # upsample 3
    with tf.variable_scope('upsample3'):
        upsample3_1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample')(up_conv2_2)
        upsample3 = tf.layers.conv2d(upsample3_1, 64, 2, padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name='up_conv3')
        mergeu3_c2 = tf.concat(values=[conv2_2, upsample3], axis=3, name='mergeu3_c2')
        up_conv3_1 = tf.layers.conv2d(inputs=mergeu3_c2, filters=64, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv6_1')
        up_conv3_2 = tf.layers.conv2d(inputs=up_conv3_1, filters=64, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv3_2')

    # upsample 4
    with tf.variable_scope('upsample4'):
        upsample4_1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample')(up_conv3_2)
        upsample4 = tf.layers.conv2d(upsample4_1, 32, 2, padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name='up_conv4')
        mergeu4_c1 = tf.concat(values=[conv1_2, upsample4], axis=3, name='mergeu7_c1')
        up_conv4_1 = tf.layers.conv2d(inputs=mergeu4_c1, filters=32, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='mergeu4_c1')
        up_conv4_2 = tf.layers.conv2d(inputs=up_conv4_1, filters=32, kernel_size=3, strides=(1, 1),
                                      activation=tf.nn.relu, padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name='up_conv4_2')
    up_conv4_3 = tf.layers.conv2d(up_conv4_2, 2, 1, padding="same",
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
    # output = tf.layers.conv2d(up_conv4_3, 1, 1, padding='same', activation=tf.nn.sigmoid,
    #                           kernel_initializer=tf.contrib.layers.xavier_initializer())

    print('output', up_conv4_3.shape)

    return up_conv4_3

