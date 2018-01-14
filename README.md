# images-like-this
----------
## The General Idea
Given a set of images, find the ones that are most similar to a set of examples. 

## More Specifically
Given a dataset, comprising image features extracted using a pre-trained model, and given also the row identifiers for a subset of the data, representing the kind of images to find, create a TRAINING set of the specified rows and a TEST set of all remaining rows, then train a binary SVM classifier to identify images in the TEST set, which are similar to those in the TRAINING set.

## Examples
`$ python3 images_like_this.py ./data/2_images/ ./data/0_training/highfive/ ./data/3_features/features_VGG16.csv --collage Yes`
