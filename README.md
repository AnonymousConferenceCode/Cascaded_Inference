# Cascaded_Inference
Cascaded Inference in Deep Neural Networks, based on softmax confidence, provides the code for reproducing the Imagenet experiments.
# Requirements
1) python 3.6
2) tensorflow 1.12
3) numpy
4) matplotlib
# Step 1 - Getting the sources
```
git clone https://github.com/AnonymousConferenceCode/Cascaded_Inference.git
git clone https://github.com/tensorflow/models.git TF_Models
cp Cascaded_Inference/Imagenet/* TF_Models/official/resnet/
cd TF_Models/official/resnet/checkpoints
mkdir clf0
mkdir clf1
mkdir clf2
wget http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v2_fp32_20181001.tar.gz
tar -xvf resnet_imagenet_v2_fp32_20181001.tar.gz
rm resnet_imagenet_v2_fp32_20181001.tar.gz
```

# Step 2 - Preprocess Imagenet 
Set the location of your imagenet tarballs (change if you need):
```
mkdir /tmp/Imagenet
set $IMAGENET_HOME = "/tmp/Imagenet"
```
