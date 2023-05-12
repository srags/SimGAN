
from autoXRD import *
from sim_gan import *

import pandas as pd
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

################################################################
# Load data and preprocess
################################################################

# Load simulated and anonimized dataset


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    theor = pd.read_csv(os.path.join(dirname, 'Datasets/theor.csv'), index_col=0)
    theor = theor.iloc[1:, ]
    theor_arr = theor.values

    # Normalize data for training
    ntheor = normdata(theor_arr)

    # Load labels for simulated data
    label_theo = pd.read_csv(os.path.join(dirname, 'Datasets/label_theo.csv'), header=None, index_col=0)
    label_theo = label_theo[1].tolist()

    # Load experimental data as dataframe
    exp_arr_new = pd.read_csv(os.path.join(dirname, 'Datasets/exp.csv'), index_col=0)
    exp_arr_new = exp_arr_new.values

    # Load experimental class labels
    label_exp = pd.read_csv(os.path.join(dirname, 'Datasets/label_exp.csv'), index_col=0).values
    label_exp = label_exp.reshape([len(label_exp), ])

    # Load class enconding
    space_group_enc = pd.read_csv(os.path.join(dirname, 'Datasets/encoding.csv'), index_col=0)
    space_group_enc = list(space_group_enc['0'])

    # Normalize experimental data
    nexp = normdata(exp_arr_new)

    # TODO: Figure out why the lengths are different and how to avoid randomly cropping it
    nexp_cropped = nexp[0:1200]
    exp_data = []
    for row in nexp_cropped:
        exp_data.append(row)

    # print(np.shape(np.array(exp_data).T))

    noisy_exp = np.array(exp_data).T


    # # PLOTTING EXP DATE PRIOR TO PREPROCESSING
    # fig, axs = plt.subplots(2, 2)
    # fig.suptitle('Experimental data before preprocessing')
    # arr1 = []
    # arr2 = []
    # arr3 = []
    # arr4 = []
    # for i in range(1499):
    #     arr1.append(nexp[i][11])
    #
    # for i in range(1499):
    #     arr2.append(nexp[i][36])
    #
    # for i in range(1499):
    #     arr3.append(nexp[i][53])
    #
    # for i in range(1499):
    #     arr4.append(nexp[i][71])
    #
    # axs[0, 0].plot(arr1)
    # axs[0, 1].plot(arr2)
    # axs[1, 0].plot(arr3)
    # axs[1, 1].plot(arr4)
    # plt.show()

    # print(nexp.shape)
    # print(nexp[0])

    # Define spectral range for data augmentation
    exp_min = 0
    exp_max = 1200
    theor_min = 125

    # window size for experimental data extraction
    window = 20
    theor_max = theor_min + exp_max - exp_min

    # Preprocess experimental data
    post_exp = normdatasingle(exp_data_processing(nexp, exp_min, exp_max, window))

    ################################################################
    # Perform data augmentation
    ################################################################

    # Specify how many data points we augmented
    th_num = 2400
    exp_aug_num = 600
    # Augment data, this may take a bit
    augd, pard, crop_augd = augdata(ntheor, th_num, label_theo, theor_min, theor_max)
    exp_aug, par = exp_augdata(post_exp, exp_aug_num, 2)
    # Enconde theoretical labels
    label_t = np.zeros([len(pard), ])
    for i in range(len(pard)):
        label_t[i] = space_group_enc.index(pard[i])

    # Input the num of experimental data points
    exp_num = 88
    # exp_aug, par = exp_augdata(noisy_exp.T, exp_aug_num, 2)
    print(np.shape(exp_aug.T))
    # Prepare experimental arrays for training and testing
    X_exp = exp_aug #np.transpose(post_exp[:, 0:exp_num])
    y_exp = label_exp[0:exp_num]

    #USING THE PREPROCESSED DATA
    # X_exp = noisy_exp #np.zeros((1200, 600)) + 0.5 #exp_aug #noisy_exp
    # Prepare simulated arrays for training and testing
    X_th = np.transpose(crop_augd)
    y_th = label_t
    print(X_th[20])
    print(X_th.shape)
    print(X_exp.shape)
    print(y_th.shape)
    print(y_exp.shape)
    print("data aug done")

    """
    Additional Raw Exp Data
    """

    # import os
    # import csv
    # cwd = os.getcwd()
    # path = '/Users/Shreyaa/Desktop/XRD Organized Folder' #/Campaign7-TP0-TP1/20190808-JT-A_C2p0_Cs13FA87MA0_XRD-TH_2-Theta_Omega.xy'
    # xrd_folder = os.listdir('/Users/Shreyaa/Desktop/XRD Organized Folder')
    # print(cwd)
    # exp_data = []
    # for folder in xrd_folder:
    #     if os.path.isdir(path+ '/' + folder):
    #         for xrd_file in os.listdir(path+ '/' + folder):
    #             if xrd_file[-2:] == 'xy':
    #                 file = path + '/' + folder + '/' + xrd_file
    #                 with open(file, 'r', newline='') as infile:
    #                     reader = csv.reader(infile, delimiter=' ')
    #                     xrd = []
    #                     for row in reader:
    #                         xrd.append(float(row[1]))
    #                     exp_data.append((xrd[0:1200]))
                        # if len(xrd) != 1374:
                        #     print(len(xrd))

    #
    # print(len(exp_data))
    # print(len(exp_data[3]))
    # y=np.array(exp_data)
    # print(y.shape)
    #
    # y = normdata(y)
    # print(y.shape)
    # y = np.linalg.norm(y)

    # X_exp = np.concatenate(X_exp)


    # # PLOTTING SIMULATED VS EXP DATA
    # fig, axs = plt.subplots(3, 3)
    #
    # th = np.random.randint(0,88,3)
    # axs[0,0].plot(noisy_exp[th[0]])
    # axs[0, 0].set_title('Noisy Experimental', fontsize=9)
    # axs[0,1].plot(noisy_exp[th[1]])
    # axs[0, 1].set_title('Noisy Experimental', fontsize=9)
    # axs[0, 2].plot(noisy_exp[th[2]])
    # axs[0, 2].set_title('Noisy Experimental', fontsize=9)
    #
    # exp = np.random.randint(0, 600, 3)
    # axs[1, 0].plot(X_exp[exp[0]], color='red')
    # axs[1, 0].set_title('Preprocessed Experimental', fontsize=9)
    # axs[1, 1].plot(X_exp[exp[1]], color='red')
    # axs[1, 1].set_title('Preprocessed Experimental', fontsize=9)
    # axs[1, 2].plot(X_exp[exp[2]], color='red')
    # axs[1, 2].set_title('Preprocessed Experimental', fontsize=9)
    #
    # exp = np.random.randint(0, 2400, 3)
    # axs[2, 0].plot(X_th[exp[0]], color='green')
    # axs[2, 0].set_title('Augmented Simulated', fontsize=9)
    # axs[2, 1].plot(X_th[exp[1]], color='green')
    # axs[2, 1].set_title('Augmented Simulated', fontsize=9)
    # axs[2, 2].plot(X_th[exp[2]], color='green')
    # axs[2, 2].set_title('Augmented Simulated', fontsize=9)
    # #
    # plt.subplots_adjust(hspace=0.5)
    # plt.subplots_adjust(hspace=1)
    # plt.xlabel(' ', fontsize=9)
    # plt.ylabel(' ', fontsize=9)
    # plt.show()



    # exp_augdata()
    # plt.plot([1, 2, 3, 4, 5])
    # plt.show()
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    # refiner_model_path = sys.argv[3]     if len(sys.argv) >= 4 else None
    # discriminator_model_path = sys.argv[4] if len(sys.argv) >= 5 else None
    cache_dir = os.path.join(path, 'cache')
    # TODO: get the pre-trained models to work without the weird tensorflow Container: localhost issue

    ref_path = None #os.path.join(cache_dir, 'refiner_model_pre_trained.h5')
    dis_path = None #os.path.join(cache_dir, 'discriminator_model_pre_trained.h5')


    main(X_th, X_exp, ref_path, dis_path)#refiner_model_path, discriminator_model_path)

    # main(sys.argv[1], sys.argv[2], refiner_model_path, discriminator_model_path)
