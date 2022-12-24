# import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras.optimizers import SGD
import keras
import tensorflow as tf
import random
import time

np.random.seed(1337)
np.set_printoptions(precision=6, suppress=True)
tf.random.set_seed(2021)
tf.config.run_functions_eagerly(True)

# flags
# MODEL_TYPE = "traditional" # set to "min-max" "strictly-positive-weight" are for previous failed attempts
MODEL_TYPE = "D" # "A", "B", "C", "D"

data = np.load('datasets/dust_sim_final_new.npz', allow_pickle=True)
if (MODEL_TYPE!="A"):
    SAMPLE_SIZE=5
    X_train, X_valid, X_test = data['Xo_train'], data['Xo_valid'], data['Xo_test']
    Y_train, Y_valid, Y_test = data['Yo_train'], data['Yo_valid'], data['Yo_test']
else:
    # taking one sample
    SAMPLE_SIZE=1
    X_train, X_valid, X_test = data['Xo_train'][:,0], data['Xo_valid'][:,0], data['Xo_test'][:,0]
    Y_train, Y_valid, Y_test = data['Yo_train'][:,0], data['Yo_valid'][:,0], data['Yo_test'][:,0]

NUM_TRAIN, NUM_TEST, NUM_VALID = len(X_train), len(X_valid), len(X_test) # todo use dataset to set this
BATCH_SIZE = 100 
# EPOCHS = 20 
HIDDEN_LAYERS = 3 
HIDDEN_NEURONS = 512 
STEPS_PER_EPOCH = NUM_TRAIN//BATCH_SIZE
ACTIVATION = "gelu"  # "relu"
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,  # initial learning rate
    decay_steps=STEPS_PER_EPOCH*20,
    decay_rate=1,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
delta_r = 0.01
# MIN_MAX_NEURONS = 7500
# MIN_GROUP_SIZE = 2  # should be divisible by MIN_MAX_NEURONS


def logit_fn(n):
    return np.log(tf.math.divide(n, 1 - n))

def logistic_transform_fn(n):
    return 1/(1+tf.math.exp(-n))

print(MODEL_TYPE)
outlier_frac=None
if (MODEL_TYPE=="B"):
    log_std = tf.constant(np.log(0.2), dtype=tf.float64)
    outlier_frac = tf.constant(0.05, dtype=tf.float64)
elif (MODEL_TYPE=="C"):
    log_std = tf.Variable(np.log(0.2), trainable=True, dtype=tf.float64) 
    logit_outlier_frac = tf.Variable(logit_fn(0.05), trainable=True, dtype=tf.float64) #0.05
elif (MODEL_TYPE=="D"): # model A and D
    log_std = None
    logit_outlier_frac = tf.Variable(logit_fn(0.05), trainable=True, dtype=tf.float64) # TODO: set to last value predicted
else: # model A
    log_std=None
    logit_outlier_frac=None
# use logstd when training on sigma model, use log_std tf variable when training on mean model

@tf.function
def custom_loss_fn(y_true, y_pred, logstd=None, outlierfrac=outlier_frac):
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    std = tf.exp(log_std) if logstd == None else tf.exp(logstd)
    std = tf.cast(std, tf.float64)
    penalty = 2 * log_std if logstd == None else 2 * logstd
    Amin, Amax = 0., 20.  # min and max range of expected A values
    # uniform outlier (bad) model
    # L_b = 1/(A_max - A_min) = 1/(20 - 0) = 0.05
    L_b = tf.cast(tf.math.log(1. / (Amax - Amin)), tf.float64)

    loss = 0
    
    # outerlierfrac is set and is constant if it is not None
    outlierfrac = logistic_transform_fn(logit_outlier_frac) if MODEL_TYPE!="B" else outlier_frac
    
    for i in range(BATCH_SIZE):
        scatter = tf.square((y_true[i] - y_pred[i]) / std)
        normalPDF = -0.5 * (tf.cast(scatter, tf.double) +
                            tf.cast(penalty, tf.double))  # discarding constants
        # inlier (good) model, averaging over normalPDF values
        L_g = tf.cast(tf.reduce_logsumexp(normalPDF), tf.float64)
        # weight L_g (good) by 1-f_out
        temp1 = tf.cast(tf.math.log(1. - outlierfrac), tf.float64) + L_g
        temp2 = tf.cast(tf.math.log(outlierfrac), tf.float64) + L_b

        L_tot = tf.reduce_logsumexp([temp1, temp2])
        loss += L_tot
    return -tf.cast(loss, tf.float64)
    
@tf.function
def mse_loss_fn(y_true, y_pred):
    # to resolve type mismatch
    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)
    loss = 0
    for i in range(BATCH_SIZE):
        loss += tf.square(y_true[i] - y_pred[i]) 
    tf.cast(loss, tf.float64)
    return loss / BATCH_SIZE


@tf.function
def train_step(x_batch_train, y_batch_train, model, optimizer=optimizer, std_vals=None):
    '''Return train loss for a training X and Y batch
    variable_sigma is True for model C

    if given std_vals for that x_batch then do not train variable sigma
    if given outlierfrac then use that instead of outlierfrac as a variable
    '''
    # open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation
    with tf.GradientTape() as tape:
        x_batch_train = tf.reshape(
            x_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 3])

        logits = model(x_batch_train, training=True)
        logits = tf.reshape(logits, [BATCH_SIZE, SAMPLE_SIZE])

        if (MODEL_TYPE=="A"):
            loss_value = mse_loss_fn(y_batch_train, logits)
        else: 
            # std vals from sigma model - not using variable sigma
            loss_value = custom_loss_fn(y_batch_train, logits) if std_vals==None \
                else custom_loss_fn(y_batch_train, logits, std_vals)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # ignore the rest if model A or B
    if (MODEL_TYPE=="A" or MODEL_TYPE=="B"): return loss_value
    
    # if std_vals is given we have spatially variable sigma std_vals
    if std_vals==None:
        with tf.GradientTape() as tape:
            x_batch_train = tf.reshape(
                x_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 3])
            logits = model(x_batch_train, training=True)
            logits = tf.reshape(logits, [BATCH_SIZE, SAMPLE_SIZE])
            loss_value = custom_loss_fn(y_batch_train, logits)  # REVIEW

        grads = tape.gradient(loss_value, [log_std])  # REVIEW
        optimizer.apply_gradients(zip(grads, [log_std]))

    # variable outlier fraction
    with tf.GradientTape() as tape:
        x_batch_train = tf.reshape(
            x_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 3])
        logits = model(x_batch_train, training=True)
        logits = tf.reshape(logits, [BATCH_SIZE, SAMPLE_SIZE])
        loss_value = custom_loss_fn(y_batch_train, logits) if std_vals==None \
            else custom_loss_fn(y_batch_train, logits, std_vals)

    grads = tape.gradient(loss_value, [logit_outlier_frac])  # REVIEW
    optimizer.apply_gradients(zip(grads, [logit_outlier_frac]))

    return loss_value


@tf.function
def val_step(x_batch_valid, y_batch_valid, model, std_vals=None):
    '''Return validation loss for a validation X and Y batch'''

    x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * SAMPLE_SIZE, 3])
    val_logits = model(x_batch_valid, training=False)
    val_logits = tf.reshape(val_logits, [BATCH_SIZE, SAMPLE_SIZE])
    
    if (MODEL_TYPE=="A"):
        loss_value = mse_loss_fn(y_batch_valid, val_logits)
    else:
        loss_value = custom_loss_fn(y_batch_valid, val_logits) if std_vals==None \
            else custom_loss_fn(y_batch_valid, val_logits, std_vals)

    return loss_value


def get_NN_pred(model, X_data):
    '''Return (A_mean, A_std) prediction given list of (x, y, z)'''
    X_data_flattened = X_data.reshape([len(X_data) * SAMPLE_SIZE, 3])
    pred = model(X_data_flattened, training=False)
    pred_np = pred.numpy()
    pred_np = pred_np.reshape([len(X_data), SAMPLE_SIZE])
    return pred_np


def get_mean_and_std_predictions(model_A, model_std, X_data):
    '''Return (A_mean, A_std) prediction given list of (x, y, z)'''
    # getting mean predictions
    X_data_flattened = X_data.reshape([len(X_data) * SAMPLE_SIZE, 3])
    pred_A = model_A(X_data_flattened, training=False).numpy()
    pred_A = pred_A.reshape([len(X_data), SAMPLE_SIZE])

    # getting std predictions
    pred_A_reshaped = pred_A.reshape([len(X_data), SAMPLE_SIZE, 1])
    std_NN_inps = np.append(X_data_flattened, pred_A_reshaped, axis=1)
    pred_std = model_std(std_NN_inps, training=False).numpy()
    return pred_A, pred_std


def get_model(A_model=False, std_model=False):
    '''Return 3 layer NN if A_model else get a 1 layer NN if std_model'''

    normalizer = preprocessing.Normalization(name="norm")

    # normalize without sampling
    if (MODEL_TYPE=="A"):
        xnormalize = X_train[:, 0]
        ynormalize = X_train[:, 1]
        znormalize = X_train[:, 2]
    # normalization with sampling - take average of x, y, z of each sample group
    else:
        xnormalize = [np.average(sample_group[:, 0]) for sample_group in X_train]
        ynormalize = [np.average(sample_group[:, 1]) for sample_group in X_train]
        znormalize = [np.average(sample_group[:, 2])
                    for sample_group in X_train]  # (6000, 1)
    

    if std_model:
        # >>> normalization without sampling
        # Anormalize = Y_train #normalize mean dust
        # <<<

        # normalizing A_mean as well if making model std?
        Anormalize = [np.average(sample_group)
                      for sample_group in Y_train]  # normalize mean dust
        tonormalize = [[x, y, z, meanA] for (x, y, z, meanA) in zip(
            xnormalize, ynormalize, znormalize, Anormalize)]  # (6000, 3)
    else:
        tonormalize = [[x, y, z] for (x, y, z) in zip(
            xnormalize, ynormalize, znormalize)]  # (6000, 3)
    normalizer.adapt(tonormalize)

    num_inps = 3 if A_model else 4
    inputs = keras.Input(shape=[num_inps, ])
    x = normalizer(inputs)

    if (std_model):
        x = layers.Dense(1024, activation=ACTIVATION, name="dense_1")(x)
    elif (A_model):
        x = layers.Dense(
            HIDDEN_NEURONS, activation=ACTIVATION, name="dense_1")(x)
        x = layers.Dense(
            HIDDEN_NEURONS, activation=ACTIVATION, name="dense_2")(x)
        x = layers.Dense(
            HIDDEN_NEURONS, activation=ACTIVATION, name="dense_3")(x)
    outputs = layers.Dense(1, activation=ACTIVATION, name="predictions")(
        x)  # activation is linear if not specified
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

################################################### MEAN MODEL ###########################################################

def train_mean_model(model_A=get_model(A_model=True),
                     # pass vals from previous trained models
                     val_loss=[], train_loss=[], std_vals=[], outlier_frac_vals=[],
                     optimizer=optimizer):
    '''
    Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    by default creates new model if none is given
    '''
    # prepare training and validation sets
    # shuffle validation and training indecies to randomize batching for Xo and Yo
    train_ind, valid_ind = [i for i in range(NUM_TRAIN)], [i for i in range(
        NUM_VALID)]  # to get same shuffle indecies for X and Y
    random.shuffle(train_ind)
    random.shuffle(valid_ind)
    # choose the correct dataset based on error parameter and batch
    X_train_batched = tf.data.Dataset.from_tensor_slices(
        [X_train[i] for i in train_ind]).batch(BATCH_SIZE)  # batch_size, 6000, 3
    Y_train_batched = tf.data.Dataset.from_tensor_slices(
        [Y_train[i] for i in train_ind]).batch(BATCH_SIZE)
    X_valid_batched = tf.data.Dataset.from_tensor_slices(
        [X_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
    Y_valid_batched = tf.data.Dataset.from_tensor_slices(
        [Y_valid[i] for i in valid_ind]).batch(BATCH_SIZE)

    with open('modelC_mean_out.txt', 'w') as f:
        epoch = 0
        # training - if last 50 validation losses, outlier fractions and std values are within 1 from each other stop
        while (epoch < 500):
        # while (epoch < 50 or max(val_loss[-50:])-min(val_loss[-50:]) > 1)\
        #         and epoch < 500\ # at least 50 epochs, at most 500
        #         and (epoch < 50 or max(std_vals[-50:])-min(std_vals[-50:]) > 0.001)\
        #         and (epoch < 50 or max(outlier_frac_vals[-50:])-min(outlier_frac_vals[-50:]) > 0.001):
            print(epoch)
            epoch += 1

        # for epoch in range(EPOCHS):
            start_time = time.time()
            print("\nStart of epoch %d" % (epoch,), file=f)

            # iterate over batches - note: x_batch_train has shape (BATCH_SIZE * SAMPLE_SIZE * 2) and y_batch_train has shape (BATCH_SIZE)
            for step, (x_batch_train, y_batch_train) in enumerate(zip(X_train_batched, Y_train_batched)):

                loss_value = train_step(
                    x_batch_train, y_batch_train, model_A, optimizer=optimizer)
                # log every 10 batches - note: for training we have 60 batches and for validation we have 20 batches
                # print("STEP: ", step, len(X_train_batched), len(Y_train_batched))
                # print(loss_value)
                if (step % 100 == 0):
                    print("Training loss (for one batch) at step %d: %.4f" %
                          (step, float(loss_value)), file=f)
                    print("Seen so far: %s samples" %
                          ((step + 1) * BATCH_SIZE), file=f)

                    # variable std and variable outlier frac predictions 
                    if MODEL_TYPE!="A" and MODEL_TYPE!="B":
                        outlier_frac = logistic_transform_fn(
                            logit_outlier_frac).numpy()
                        
                        # prediction for sigma as free parameter
                        print("std prediction: ", np.exp(log_std), file=f)
                        print("outlier frac prediction: ", outlier_frac, file=f)
                        
                        # saving variable outlier fraction and sigma >>>
                        std_vals.append(np.exp(log_std))  # add the final std prediction
                        outlier_frac_vals.append(outlier_frac)

            # run validation loop at the end of each epoch.
            for (x_batch_valid, y_batch_valid) in zip(X_valid_batched, Y_valid_batched):
                val_loss_value = val_step(
                    x_batch_valid, y_batch_valid, model_A)

            # save model if better than best model
            if (len(val_loss) >= 2 and val_loss_value < val_loss[-1]):
                model_A.save('model_mean_best')
            # save most recent model
            model_A.save('model_mean_recent')

            train_loss.append(loss_value)  # save train loss for plotting
            val_loss.append(val_loss_value)  # save val loss for plotting   
            
            print("Time taken: %.2fs" % (time.time() - start_time), file=f)

    return model_A, train_loss, val_loss, std_vals, outlier_frac_vals


################################################### SIGMA MODEL ##########################################################
@tf.function
def train_step_for_std_model(model_A, model_std, x_batch_train, y_batch_train, constant_log_std, optimizer=optimizer):
    '''Return train loss for a training X and Y batch when completing the training step for both models
    constant_log_std and constant_outlier_frac'''
    with tf.GradientTape() as tape:
        x_batch_train = tf.reshape(
            x_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 3])  # (1000, 3)

        logits_A = model_A(x_batch_train, training=False)
        logits_A = tf.reshape(logits_A, [BATCH_SIZE, SAMPLE_SIZE])

        y_batch_train_for_std = tf.reshape(
            logits_A, [BATCH_SIZE * SAMPLE_SIZE, 1])  # (1000, 1)
        x_batch_train_for_std = np.append(
            x_batch_train, y_batch_train_for_std, axis=1)  # (1000, 4)

        # [BATCH_SIZE * SAMPLE_SIZE, 1]
        # logits_std = model_std(x_batch_train_for_std, training=True)
        logits_std = get_std_model_pred(model_std,
                                        x_batch_train_for_std,
                                        constant_log_std,
                                        True)

        logits_std = tf.cast(logits_std, tf.float64)
        train_loss_std_val = custom_loss_fn(y_batch_train, logits_A, logits_std)  # std

    grads = tape.gradient(train_loss_std_val, model_std.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_std.trainable_weights))

    # train mean model
    # after training the model_std, train the mean model with the sigma prediction for each x_batch_train
    # given custom_loss the logstd value that is predicted by the sigma model
    train_loss_A_val = train_step(x_batch_train, y_batch_train, model_A,
                                  optimizer=optimizer, std_vals=logits_std)  # give std vals if dealing with spatially varying sigma
    return train_loss_A_val, train_loss_std_val


@tf.function
def val_step_for_std_model(model_A, model_std, x_batch_valid, y_batch_valid, constant_log_std):
    x_batch_valid = tf.reshape(x_batch_valid, [BATCH_SIZE * SAMPLE_SIZE, 3])
    val_logits_A = model_A(x_batch_valid, training=False)
    val_logits_A = tf.reshape(val_logits_A, [BATCH_SIZE, SAMPLE_SIZE])

    y_batch_valid_for_std = tf.reshape(
        val_logits_A, [BATCH_SIZE * SAMPLE_SIZE, 1])  # (1000, 1)
    x_batch_valid_for_std = np.append(
        x_batch_valid, y_batch_valid_for_std, axis=1)  # (1000, 4)

    val_logits_std = get_std_model_pred(model_std,
                                        x_batch_valid_for_std,
                                        constant_log_std,
                                        False)

    val_logits_std = tf.cast(tf.math.reduce_mean(val_logits_std), tf.float64)
    val_loss_std_val = custom_loss_fn(y_batch_valid, val_logits_A, val_logits_std)

    # validation for mean model after getting val logits for sigma values
    val_loss_A_val = val_step(x_batch_valid, y_batch_valid, model_A, val_logits_std)

    return val_loss_A_val, val_loss_std_val


def get_std_model_pred(model_std, x, constant_log_std, training=False):
    return model_std(x, training=training) * constant_log_std


def train_sigma_model(model_A, constant_log_std, model_std=get_model(std_model=True), 
                      val_loss_A=[], val_loss_std=[],
                      train_loss_std=[], train_loss_A=[],
                      outlier_frac_vals=[],
                      optimizer=optimizer):
    '''Source: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch'''
    # prepare training and validation sets
    # shuffle validation and training indecies to randomize batching for Xo and Yo
    train_ind, valid_ind = [i for i in range(NUM_TRAIN)], [i for i in range(
        NUM_VALID)]  # to get same shuffle indecies for X and Y
    random.shuffle(train_ind)
    random.shuffle(valid_ind)
    # choose the correct dataset based on error parameter and batch
    X_train_batched = tf.data.Dataset.from_tensor_slices(
        [X_train[i] for i in train_ind]).batch(BATCH_SIZE)  # batch_size, 6000, 3
    Y_train_batched = tf.data.Dataset.from_tensor_slices(
        [Y_train[i] for i in train_ind]).batch(BATCH_SIZE)
    X_valid_batched = tf.data.Dataset.from_tensor_slices(
        [X_valid[i] for i in valid_ind]).batch(BATCH_SIZE)
    Y_valid_batched = tf.data.Dataset.from_tensor_slices(
        [Y_valid[i] for i in valid_ind]).batch(BATCH_SIZE)

    # with open('out_std.txt', 'w') as f:
    for epoch_i in range(500):
        print('done ', str(epoch_i))
        start_time = time.time()
        print("\nStart of epo ch %d" % (epoch_i + 1,))#, file=f)

        for step, (x_batch_train, y_batch_train) in enumerate(zip(X_train_batched, Y_train_batched)):
            train_loss_A_val, train_loss_std_val = train_step_for_std_model(
                model_A, model_std, x_batch_train, y_batch_train, constant_log_std, optimizer=optimizer)
            if (step % 100 == 0):
                print("Training loss (for one batch) at step %d: A: %.4f std:%.4f" %
                    (step, float(train_loss_A_val), float(train_loss_std_val)))#, file=f)
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE)) #, file=f)

                x_batch_train_for_std = tf.reshape(
                    x_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 3])  # (1000, 3)
                y_batch_train_for_std = tf.reshape(
                    y_batch_train, [BATCH_SIZE * SAMPLE_SIZE, 1])  # (1000, 1)


                x_batch_train_for_std = np.append(
                    x_batch_train_for_std, y_batch_train_for_std, axis=1)  # (1000, 4)

        # run validation loop at the end of each epoch.
        for (x_batch_valid, y_batch_valid) in zip(X_valid_batched, Y_valid_batched):
            val_loss_A_val, val_loss_std_val = val_step_for_std_model(
                model_A, model_std, x_batch_valid, y_batch_valid, constant_log_std)
        
        # IDEA saving model >>>
        # save model if better than best model
        if (len(val_loss_std) >= 2 and val_loss_std_val < val_loss_std[-1]):
            model_std.save('modelstd_best')
            model_A.save('modelA_best')
        # save most recent model
        # model_std.save('modelstd_recent')        
        # model_A.save('modelA_recent')
        
        # every 10 epochs save model
        if (epoch_i % 10 == 0):
            model_A.save('epoch_' + str(epoch_i) + '_model_mean')
            model_std.save('epoch_' + str(epoch_i) + '_model_std')
        # IDEA saving model <<<
        
        train_loss_A.append(train_loss_A_val)
        train_loss_std.append(train_loss_std_val)
        val_loss_A.append(val_loss_A_val)  
        val_loss_std.append(val_loss_std_val) 
        outlier_frac = logistic_transform_fn(logit_outlier_frac).numpy()
        outlier_frac_vals.append(outlier_frac)

        print("Time taken: %.2fs" % (time.time() - start_time)) #, file=f)
    # >FIXME add defn for test_pred after fixing the speed issue
    #test_pred = get_NN_pred(model, Xo_samp_test, error) if error else get_NN_pred(model, X_test, error)
    return model_std, model_A, train_loss_A, train_loss_std, val_loss_A, val_loss_std, outlier_frac_vals


###########################################################################################################################
###############################################   FAILED ATTEMPS    #######################################################
###########################################################################################################################
# @tf.function
# def loss_fn(y_true, y_pred, train=False, error=False):
#     '''Return loss for data with error bars in both X and Y if error=True else return loss for data with no errors
#     y_true has shape (BATCH_SIZE); its a batch either from Yo_train or Yo_valid
#     y_pred has shape (BATCH_SIZE * SAMPLE_SIZE); its prediction of model when given a batch of X that has SAMPLE_SIZE samples per item
#     Set train to true if given a training set, else if given a set validation set, set train to false'''
#     # to resolve type mismatch
#     y_true = tf.cast(y_true, tf.double)
#     y_pred = tf.cast(y_pred, tf.double)
#     loss = 0
#     if not error:  # no errors in either X or Y
#         for i in range(BATCH_SIZE):
#             loss += tf.square(y_true[i] - y_pred[i])
#         tf.cast(loss, tf.float64)
#         return loss / BATCH_SIZE
#     # errors in both X and Y
#     y_true = tf.cast(y_true, tf.double)
#     y_pred = tf.cast(y_pred, tf.double)
#     # reshape y_pred to (BATCH_SIZE, SAMPLE_SIZE)
#     y_pred = tf.reshape(y_pred, [BATCH_SIZE, SAMPLE_SIZE])
#     # set errors to Ye_train or Ye_valid based on flag
#     Y = Y_train if train else Y_valid
#     for i in range(BATCH_SIZE):
#         for j in range(SAMPLE_SIZE):
#             # calculate lin or asinh difference
#             diff = tf.math.subtract(y_true[i], y_pred[i][j])
#             diff = tf.cast(diff, tf.float64)
#             loss += tf.math.divide(tf.square(diff), np.square(Y[i]))
#     return loss / (BATCH_SIZE * SAMPLE_SIZE)


# class My_Init(tf.keras.initializers.Initializer):
#     '''Initializes weight tensors to be non negative and have mean and standard dev given'''

#     def __init__(self, mean, stddev):
#         self.mean = mean
#         self.stddev = stddev

#     def __call__(self, shape, dtype=None):
#         initializers = np.random.normal(
#             self.mean, self.stddev, size=shape)  # get normalized random data
#         initializers = initializers.astype("float32")
#         # keep the weights from the input layer to the 1st hidden layer for input theita the same
#         # update negative values to positive value 1e-10
#         initializers[initializers < 0] = 0.
#         # set theita weights back
#         # make sure this is the first layer of weights
#         if len(initializers) == 2 or len(initializers) == 3:
#             # generate weights for theita input neuron (how they are generated by default)
#             initializers[1] = np.random.normal(
#                 0., 0.05, size=len(initializers[1]))
#         return tf.convert_to_tensor(initializers)  # convert to tensor


# class My_Constraint(tf.keras.constraints.Constraint):
#     '''Constrains weight tensors to be non negative'''

#     def __call__(self, w):
#         w_np = w.numpy()
#         # keep the weights from the input layer to the 1st hidden layer for input theita the same
#         if len(w_np) == 2 or len(w_np) == 3:  # make sure this is the first layer of weights
#             # make copy of weights connected to theita
#             theita_w_copy = np.copy(w[1])
#             # >FIXME
#         # update negative values to positive value 0.
#         w_np[w_np < 0] = 0.
#         # set theita weights back
#         if len(w_np) == 2 or len(w_np) == 3:
#             w_np[1] = theita_w_copy
#         return tf.convert_to_tensor(w_np)


# def get_strictly_positive_weight_NN_model(error=False, num_input=2):
#     '''Return strictly positive weight model
#     >FIXME FAILED ATTEMPT (explain why)'''
#     # fit state of preprocessing layer to data being passed
#     # ie. compute mean and variance of the data and store them as the layer weights
#     # preprocessing.Normalization(input_shape=[2,], dtype='double')
#     normalizer = preprocessing.Normalization()
#     normalizer.adapt([np.average(x_obs) for x_obs in X_train]
#                      ) if error else normalizer.adapt(X_train)
#     inputs = keras.Input(shape=[num_input, ])
#     x = normalizer(inputs)
#     for i in range(HIDDEN_LAYERS):
#         x = layers.Dense(HIDDEN_NEURONS, activation=ACTIVATION, name="dense_" + str(
#             i + 1), kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
#     # activation is linear if not specified
#     outputs = layers.Dense(1, name="predictions", kernel_initializer=My_Init(
#         2., 1.), kernel_constraint=My_Constraint())(x)
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def get_partial_min_max_model(num_input=2, error=True):
#     '''Returns 2 layer NN model with input layer of size num_input and output layer of size MIN_MAX_NEURONS 
#     >FIXME FAILED ATTEMPT (explain why)'''
#     normalizer = preprocessing.Normalization()
#     normalizer.adapt([np.average(x_obs) for x_obs in X_train]
#                      ) if error else normalizer.adapt(X_train)
#     inputs = keras.Input(shape=[num_input, ])
#     x = normalizer(inputs)
#     # outputs are MIN_MAX_NEURONS lengthed
#     outputs = layers.Dense(MIN_MAX_NEURONS, activation="softplus", name="outputs",
#                            kernel_initializer=My_Init(2., 1.), kernel_constraint=My_Constraint())(x)
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def get_min_max_model_predictions(model, x_batch, training):
#     '''if no_batching=False we are dealing with training and validation data that have batches, this needs additional steps in reshaping the tensors
#     if no_batching=True we are dealing with testing data that do not have batches'''
#     logits = model(x_batch, training=training)
#     # >FIXME change the name of the x_batch to take into account getting predictions for the testing set too
#     # instead of MIN_MAX_NEURONS * BATCH_SIZE // 2 we use len(x_batch) since for testing the len(x_batch) is different than trianing or validation
#     reshaped = tf.reshape(logits, [
#                           MIN_MAX_NEURONS * len(x_batch) // MIN_GROUP_SIZE, MIN_GROUP_SIZE])  # (1500000, 2)
#     apply_min = tf.reduce_min(reshaped, axis=1)  # (1500000, )
#     apply_min_reshaped = tf.reshape(
#         apply_min, [len(x_batch), MIN_MAX_NEURONS // MIN_GROUP_SIZE])  # (1500000, )
#     logits_minmax = tf.reduce_max(apply_min_reshaped, axis=1)

#     return logits_minmax
