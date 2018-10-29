import os
import tensorflow as tf
import numpy as np
import skimage.io
import scipy.misc as misc
import pickle
from utils import *
#from DRAGAN import *
from ACGAN import *
#from WGAN_GP import *

def main():
    print("Loading data...")
    if is_preprocsee:
        img_feat, tags_idx = preprocessing(prepro_dir, img_dir, tag_path)
    else:
        img_feat = pickle.load(open(os.path.join(prepro_dir, "img_feat.dat")))
        tags_idx = pickle.load(open(os.path.join(prepro_dir, "tags.dat")))
    print("Normalizing...")        
    img_feat = np.array(img_feat, dtype='float32')/127.5 - 1. #normalize
    

    data = Data(img_feat, tags_idx, noise_dim)
    data.load_eval(test_path)
    print("Building Network...")
    model = ACGAN(data)
    #model = DRAGAN(data)
	#model = WGAN_GP(data, vocab_processor)
    model.build_model()
    print("Training...")
    model.train()

if __name__ == '__main__':
    is_preprocsee = True
    
#    save_path = '../model/'
#    load_path = '../model/'
    test_path = '../data/testing_tags.txt'
    output = '../result/'
    tag_path = '../data/tags_clean.csv'
    img_dir = '../data/faces'
    prepro_dir = '../model/prepro/'
    test_tags_idx = ""
    
    noise_dim = 100
    main()
    
