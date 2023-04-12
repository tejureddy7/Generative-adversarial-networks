README.md
-------------------------

This folder contains artifacts related to the final project submission of Vineeth NC, Tejaswini Manjunath, and Sai Charita Thati for CMSC 678 - Introduction to Machine Learning, Spring 2022. The project deals with using GANs for data augmentation for image classifiers.
https://drive.google.com/drive/folders/1r6LGINxsPD_pfNdMceWEeXQkEGw_NfEM
The artifact zip directory structure is as follows:

* CODE
* DATASET
* MODELS
* RESULTS
  * 1-N0GAN
  * 2-1000
  * 3-JUST GAN
  * 4-GAN+ORIGINAL
  * 5-newGanImages
  * 6-FullGan
  * 7-GAN_Training
  * 8-DA_Results



CODE
-------------------------
Contains different codes that are used for the project.

GAN_final_code.ipynb is a Jupyter notebook that can be used to train the GAN and generate images for a specific class.
kaggle.json is a helper file that is used by the GAN training notebook to fetch the dataset from the internet.
Run baseline.py to compare different augmentation datasets.
Specific instructions are written in the file.



DATASET
-------------------------
This folder contains datasets to be used for training the classfier.

archive.zip is the original dataset from kaggle which contains 3 folds for training, testing andd validation.
ganseg_combined.zip contains 12000 GAN generated images, 2000 for each class.
Full.zip contains original data for training as well as data generated using GANS.



MODELS
-------------------------
Contains the weights of the generator and discriminators of different class GANs. 
The files with the suffix "_gen" refer to the generator weights, while "_dis" refer to the discriminator weights.



RESULTS
-------------------------
Folders 1 through 6 contain training results and weight files for all Classifier experiments.
Folder 7 contains the graphs of the losses of the training of GAN across different classes.
Folder 8 contains the results of the data augmentations.
# Generative-adversarial-networks
