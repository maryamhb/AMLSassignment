from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from models.lenet import LeNet
import classification as cls
import numpy as np
import utils as ut
import time
import cv2
import os


# Get data and run binary classifiers
# Store predictions, stats and images
def test_binary():
    # Initialise lists
    pred_list = []
    models_list = []
    conf_m_list = []
    cv_score_list = []

    it_range = [0, 1]

    # Define model
    te_model = 'Test'
    arg = 'lbfgs'
    all_i, all_x, all_y = cls.get_data('HoG')

    tr_x, tr_y, te_nom, te_x, te_y = cls.split_data(all_i, all_x, all_y)

    # Map columns to tasks
    tasks = {0: 5, 1: 3, 2: 1, 3: 2, 4: 4}

    init = time.time()

    # Classify task by task
    for col in range(it_range[0], it_range[1]):
        t_num = tasks[col]
        print('\nTask', t_num, '-')

        # Train model classifier
        model, pred, c_matrix = cls.train_mlp(arg, tr_x, tr_y[:, col],
                                          te_x, te_y[:, col])

        # 10-fold cross validation
        print('Cross validating')
        # 10-fold cross validation score
        cv_score = cls.cross_val(te_model, all_x, all_y[:, col], 10, False, model)

        # Write to lists for later
        print('Storing results')
        # Report performance in csv
        models_list.append(model)
        conf_m_list.append(c_matrix)
        pred_list.append(pred)
        cv_score_list.append(cv_score)

    end = time.time()

    # Store results in .csv
    print('\nReporting performance')
    for col in range(it_range[0], it_range[1]):
        print('Task', tasks[col])
        ut.report_binary(te_model, arg, tasks[col], te_nom,
                       pred_list[col-1], conf_m_list[col-1], cv_score_list[col-1])

    # Plot learning curves
    print('\nPlotting graphs')
    for col in range(it_range[0], it_range[1]):
        print('Task', tasks[col])
        fig = ut.plot_learning_curve(models_list[col-1], te_model + ' ' + arg + ' Learning curve',
                                     all_x, all_y[:, col], (0.4, 1.01), cv=10, n_jobs=4)
        fig.savefig(os.path.join('out', 'Graphs', 'T' + str(tasks[col]) + '_' + te_model + '_' + arg + '.png'))

    ut.report_time(tasks[it_range[0]], te_model, arg, end - init)

    print(end - init)

    return 0


# Get data, run multi-class classifiers
# Store predictions, stats and images
def test_multiclass():
    # initialise lists
    all_i = []
    all_x = []
    all_y = []
    count = 0

    # define model
    te_model = 'SVM'
    arg = 'rbf'

    print('\nLoading data')
    img_paths, img_labels = ut.load_data(1, 2)

    if os.path.isdir(ut.img_dir):

        for img_path in img_paths:
            img_name = img_path.split('.')[2].split('/')[-1]

            img = image.img_to_array(image.load_img(img_path,
                                                    target_size=None,
                                                    interpolation='bicubic'))
            # halve image size x3 to reduce complexity
            img = cv2.resize(img, (32, 32))

            all_i.append(img_name)
            all_x.append(img)
            all_y.append(img_labels[img_name])

            count += 1
            #if count > 100: break

    imgs = np.array(all_x)
    labels = np.array(all_y).ravel()

    # reshape images to 2D matrix
    m = imgs.shape
    imgs = np.reshape(imgs, (m[0], m[1] * m[2] * 3)).astype(float)

    # split dataset into training and test
    tr_x, tr_y, te_nom, te_x, te_y = cls.split_data(all_i, imgs, labels)

    # remove noise from training data using HoG
    print('Removing noise')
    tr_xx, tr_yy = ut.denoise_training(all_i[0:len(tr_x)], tr_x, tr_y)

    init = time.time()

    # train model classifier
    print('Training model')
    model, pred, c_matrix = cls.train_svm(arg, tr_xx, tr_yy,
                                          te_x, te_y)
    end = time.time()

    # store results in .csv
    print('Reporting performance')
    ut.report_multiclass(te_model, arg, te_nom,
                       pred, c_matrix)
    ut.report_time(5, te_model, arg, end - init)
    print(end - init)

    return 0


# Get data, run multi-class classifiers
# Store predictions, stats and images
def test_deeplearn():
    init = time.time()

    # initialise the number of epochs, initial
    # learning rate, and batch size
    epoch_n = 25
    init_alpha = 1e-3
    batch_n = 32

    # initialise data lists
    print("Loading images")
    data = []
    names = []
    labels = []
    count = 0

    # extract processed images and labels
    img_paths, img_labels = ut.load_data(1, 2)
    if os.path.isdir(ut.img_dir):
        for img_path in img_paths:
            img_name = img_path.split('.')[2].split('/')[-1]

            img = cv2.imread(img_path)
            img = cv2.resize(img, (28, 28))
            img = image.img_to_array(img)

            data.append(img)
            names.append(img_name)
            labels.append(img_labels[img_name])
            
            count +=1
            #if count > 50: break

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

    # TODO: split data to training, validation and test
    # partition data into training and testing
    tr_x, tr_y, te_i, te_x, te_y = cls.split_data(names, data, labels)

    # remove noise from training data using HoG
    print("Denoising training set")
    tr_xx, tr_yy = ut.denoise_training(names[0:len(tr_x)], tr_x, tr_y)

    # convert the labels from integers to vectors
    tr_yy = to_categorical(tr_yy, num_classes=6)
    te_y = to_categorical(te_y, num_classes=6)

    # construct the image generator for data augmentation
    aug = image.ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, fill_mode="nearest")

    # initialise the model
    print("Compiling model")
    model = LeNet.build(width=28, height=28, depth=3, classes=6)
    print(type(model))
    opt = Adam(lr=init_alpha, decay=init_alpha / epoch_n)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("Training network")
    H = model.fit_generator(aug.flow(tr_xx, tr_yy, batch_size=batch_n),
                            validation_data=(te_x, te_y), steps_per_epoch=len(tr_xx) // batch_n,
                            epochs=epoch_n, verbose=1)

    # save the model to disk
    print("Serialising network")

    # directory to store results
    CNN_dir = os.path.join('out', 'Classification', 'CNN')
    model.save(os.path.join(CNN_dir, 'multiclass.model'))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch_n
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Accuracy on Hair Colour")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join('out', 'Graphs', 'T5_CNN_Keras.png'))

    # predict test data
    probs_list = model.predict(te_x)

    # extract labels from probabilities
    te_len = len(probs_list)
    preds = np.zeros(te_len)
    i = 0
    for prob in probs_list:
        max_prob = 0
        j = 0
        for p in prob:
            if p > max_prob:
                preds[i] = -1 if j == 0 else j
                max_prob = p
            j += 1
        i += 1

    # Turn true labels into 1D array
    te_labels = np.zeros(te_len)
    for row in range(te_len):
        for col in range(6):
            if te_y[row][col] == 1:
                te_labels[row] = col if col > 0 else -1

    # get confusion matrix
    conf_m = confusion_matrix(te_labels, preds)
    end = time.time()

    print(preds)

    # store accuracy and predictions
    ut.report_multiclass('CNN', '', te_i, preds, conf_m)
    ut.report_time(5, 'CNN', '', end - init)

    return 0


test_deeplearn()
