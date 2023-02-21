import sys
sys.path.append("../")
import os
import shutil
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.utils.np_utils import to_categorical
from sklearn import svm, datasets, preprocessing

import classification.utils_load_learnsetup
from classification.LearnSetups.LearnSetup import LearnSetup
import evasion.utils_attack as ua
from evasion.Author import Author
from train204 import Sample_Concrete, Concatenate, get_original_LearnSetup

datasetpath = '/workspace/code-imitator-master/data/dataset_2017/dataset_2017_8_formatted_macrosremoved/'


# Get examples
source_dir = '/workspace/code-imitator-master/src/explain/generate_example/'
dist_dir = '/workspace/code-imitator-master/src/explain/nochange_example/'

E = load_model('models/E.h5', custom_objects={'Sample_Concrete': Sample_Concrete, 'Concatenate': Concatenate})

originalLearnSetup = get_original_LearnSetup()
train_object = originalLearnSetup.data_final_train
x_train = train_object.getfeaturematrix().todense()
sc = preprocessing.StandardScaler().fit(x_train)

if __name__ == '__main__':
    for au in os.listdir(source_dir):
        os.makedirs(os.path.join(dist_dir, au))
        for f in os.listdir(os.path.join(source_dir, au)):
            os.makedirs(os.path.join(dist_dir, au, f))

            original_filepath = os.path.join(datasetpath, au, f+".cpp")
            original_data_merged = ua.load_new_features_merged(
                datasetpath=datasetpath,
                attackdirauth='',
                verbose=False, cppfile=original_filepath,
                train_object=originalLearnSetup.data_final_train,
                already_extracted=False)
            original_classpred = originalLearnSetup.predict(
                feature_vec=original_data_merged.getfeaturematrix()[0, :])
            original_scorepred = originalLearnSetup.predict_proba(
                feature_vec=original_data_merged.getfeaturematrix()[0, :],
                target_class=original_classpred)
            num = 0
            for df in os.listdir(os.path.join(source_dir, au, f)):
                source_file = os.path.join(source_dir, au, f, df)
                # A. Load features TODO, use feature_paths instead of already_extracted...
                attack_data_merged = ua.load_new_features_merged(
                    datasetpath=datasetpath,
                    attackdirauth='',
                    verbose=False, cppfile=source_file,
                    train_object=originalLearnSetup.data_final_train,
                    already_extracted=False)

                # B. Evaluate and update score and class of target class
                targetauthor = Author(author=au, learnsetup=originalLearnSetup)
                #print("true class:", targetauthor.true_class)
                scoreprednew = originalLearnSetup.predict_proba(
                    feature_vec=attack_data_merged.getfeaturematrix()[0, :],
                    target_class=targetauthor.true_class)

                
                # print(feature_vec)
                classprednew = originalLearnSetup.predict(
                    feature_vec=attack_data_merged.getfeaturematrix()[0, :])

                
                #print("Pred:{} /({})".format(round(scoreprednew, 4), classprednew))
                # print("Pred:{}".format(classprednew))
                if(original_classpred == classprednew): #and abs(original_scorepred - scoreprednew) < 0.2): 
                    num = num + 1
                    shutil.copy(source_file, os.path.join(dist_dir, au, f, df))

