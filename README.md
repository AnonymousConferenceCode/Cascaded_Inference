# Cascaded_Inference
This project provides the code and the instructions for reproducing the Imagenet experiments from the article "Dynamicaly Sacrificing Accuracy for Reduced Computation: Cascaded Inference based on Softmax Confidence"
# Requirements
1) python 3.6
2) tensorflow 1.12
3) numpy
4) matplotlib
# Step 1 - Getting the sources
Run the following commands to clone the sources and to download the pre-trained official Imagenet model
```
git clone https://github.com/AnonymousConferenceCode/Cascaded_Inference.git
git clone https://github.com/tensorflow/models.git TF_Models
cp Cascaded_Inference/Imagenet/* TF_Models/official/resnet/
cd TF_Models/official/resnet/checkpoints
mkdir clf0
mkdir clf1
mkdir clf2
mkdir output_softmax
wget http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v2_fp32_20181001.tar.gz
tar -xvf resnet_imagenet_v2_fp32_20181001.tar.gz
rm resnet_imagenet_v2_fp32_20181001.tar.gz
cd ../../../..
```

# Step 2 - Preprocess Imagenet 
Set the location of your imagenet tarballs using $IMAGENET_HOME (change it to your favorite dir)
```
mkdir /tmp/Imagenet
export $IMAGENET_HOME = "/tmp/Imagenet"
cp Cascaded_Inference/Imagenet_tools/* $IMAGENET_HOME 
cd $IMAGENET_HOME 
./preprocess_imagenet.sh
cd -
cd TF_Models/official/resnet
```
# Step 3 - Fine tuning the cascade (according to section 6.2 in the artice)
First, evaluate the pre-trained model (change the -num_gpus according to your machine):
```
python3 imagenet_main.py --eval_only --is_cascaded --classifier_id 2 --data_format channels_first -rv 2 -rs 50 -md checkpoints/clf2 -pmcp checkpoints/resnet_imagenet_v2_fp32_20181001 --data_dir /data/datasets/Imagenet/tf_records --num_gpus 4
```
Second, train and evaluate the classifiers $clf_0$ and $clf_1$
