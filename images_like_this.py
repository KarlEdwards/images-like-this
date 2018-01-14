#!/usr/bin/env python3

'''
Find images similar to these...

Given a dataset, comprising image features extracted using a pre-trained model, and given also the row identifiers for a subset of the data, representing the kind of images to find, create a TRAINING set of the specified rows and a TEST set of all remaining rows, then train a binary SVM classifier to identify images in the TEST set, which are similar to those in the TRAINING set.

INPUT:
  Image_file_path: a directory of image files
  Data:            a file, containing features that have been extracted from the files in the image file directory
  Exemplar_Ids   : a list of image files containing positive examples

PROCESSING:
  
OUTPUT:
  Collage:         a png file, containing a matrix of images, the first row illustrating the given exemplars, subsequent rows depicting images predicted to be similar
'''

# Example Usage:
# python3 images_like_this.py path/to/images/ path/to/exemplars/ feature/file.csv --output collage.png
# $ python3 images_like_this.py ./data/2_images/ ./data/5_exemplars/ ./data/3_features/features_VGG16.csv --output collage.png

# PREPARATION
# Copy example images images ---> exemplars
#
# for f in 1609 1612 1621 1627 2548; do   echo cp ./2_images/frame_0$f.png ./5_exemplars/; done # | sh

# -----------------------------------------------
# Load libraries
# -----------------------------------------------

import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np
import csv
import math
import re
from sklearn import svm
from square_collage import Collage

# Configuration

FEATURE_FILE='./data/3_features/features_VGG16.csv'
DEFAULT_GAMMA=0.0001
DEFAULT_C=100
RATIO_NEGATIVE_TO_POSITIVE = 1.5  ## This appears to have no effect; see Gamma, C
negative_images = []
negative_examples = []

# -----------------------------------------------
#         gamma and C
# -----------------------------------------------
# https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine

# C is the parameter for the soft margin cost function, which controls the influence of each individual support vector; this process involves trading error penalty for stability.

# large gamma leads to high bias and low variance models, and vice-versa
# -----------------------------------------------

def get_args( arg_values ):
    parser = argparse.ArgumentParser( prog = 'images_like_this' )
    parser.add_argument( 'images', help = 'base path to image files' )
    parser.add_argument( 'exemplars', help = 'path to copies of desired images' )
    parser.add_argument( 'features', help = 'file of features extracted from images' )
    parser.add_argument( '-o', '--output', help = 'output file name (collage file)' )
    parser.add_argument( '-c', '--collage', help = 'include this to show collage' )
    return parser.parse_args( arg_values[ 1: ] ) # Return everything but the program name

def show_args( args ):
    print( args )

def choose_n( df, N ):
   return df.ix[ np.random.choice( df.index, N, replace = False )]
   
def load_features( f ):
    # Get the data (features extracted from images, in this case)
    return pd.read_csv( f, header = None )

def get_image_names( data ):
    # Return only the first column, which contains image file names
    return data.iloc[ :, 0 ]

def make_collage( params, cfg ):
    collage = Collage( params )
    if cfg.output:
        collage.save_to_file( cfg.output )
    else:
        collage.show()

def classify( Xtrain, ytrain, Xtest, gamma, C ):
    classifier = svm.SVC( gamma=gamma , C=C )
    classifier.fit( Xtrain, ytrain )
    return classifier.predict( Xtest )

def make_file_specs( path, file_names ):
    return [path + s for s in file_names]

def file_list( path ):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir( path )
    return [ x for x in filelist if not ( x.startswith( '.' ) ) ]

# -----------------------------------------------
#   M A I N   P R O G R A M
# -----------------------------------------------

def main( argv ):
    cfg = get_args( argv )
    
    # -------------------------------------------
    feature_data = load_features( FEATURE_FILE )
    image_names  = get_image_names( feature_data )
    print( 'Found features for ' + str( len( image_names )) + ' images.')
    print( str( np.shape( feature_data )[1] ) + ' features per image.' )

    print( '---------------' )
    topic = re.sub(r"(.*)\/(.*)\/", r'\2', cfg.exemplars ) 
    print( 'positive_' + topic )
    print( '---------------' )
    
    # -------------------------------------------
    positive_images = pd.Series( file_list( cfg.exemplars )) 
    positive_examples  = pd.DataFrame( feature_data[  feature_data[ 0 ].isin( positive_images )])
    remaining_examples = pd.DataFrame( feature_data[ ~feature_data[ 0 ].isin( positive_images )])

    # -------------------------------------------
    # Randomly select a number of rows to represent negative examples
    # (Even though we don't know for sure they are such)
    negative_examples = choose_n( remaining_examples, int( RATIO_NEGATIVE_TO_POSITIVE * len( positive_examples )))
    negative_images = negative_examples.iloc[ :, 0 ]

    # -------------------------------------------
    print( 'Found ' + str( len( positive_images )) + ' exemplars, including:' )
    print( positive_images[:5] )
    collage = Collage( make_file_specs( cfg.images, positive_images ) )
    collage.add_title( 'Positive Examples' )
    collage.show()
    collage.save_to_file( 'positive_' + topic + '.png' )
    
    print( 'Found ' + str( len( negative_images )) + ' non-exemplars, including:' )
    print( negative_images[:5] )
    
    collage = Collage( make_file_specs( cfg.images, negative_images ) )
    collage.add_title( 'Negative Examples' )
    collage.show()
    collage.save_to_file( 'negative_' + topic + '.png' )
    
    # -------------------------------------------
    # Consolidate the TRAINING set
    Xtrain = positive_examples.append( negative_examples ).iloc[ :, 1: ]

    test_examples = remaining_examples[ ~remaining_examples[ 0 ].isin( Xtrain.iloc[ :, 0 ] )]
    test_images = np.array( test_examples.iloc[ :, 0 ] )
    Xtest = test_examples.iloc[ :, 1: ]

    training_images = positive_images.append( negative_images )
    ytrain = [ 1 ] * len( positive_examples ) + [ 0 ] * len( negative_examples )

    # -------------------------------------------
    predicted = classify( Xtrain, ytrain, Xtest, DEFAULT_GAMMA, DEFAULT_C )

    # -------------------------------------------
    if cfg.collage:
        collage = Collage( make_file_specs( cfg.images, test_images[ predicted == 1 ] ) )
        collage.add_title( 'Results' )
        collage.show()
        collage.save_to_file( 'predicted_' + topic + '.png' )
    
# -------------------------------------------
if __name__ == '__main__':
    sys.exit( main( sys.argv ))
