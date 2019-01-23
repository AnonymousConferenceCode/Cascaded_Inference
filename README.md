# Cascaded_Inference
This project provides the code and the instructions for reproducing the Imagenet experiments from the article "Dynamicaly Sacrificing Accuracy for Reduced Computation: Cascaded Inference based on Softmax Confidence". The excel file "MAC_count_calculations.xlsx" contains the analytical computations of the multiply-accumulate (MAC) operations of the ResNet models that where involved in the experiments.

## Requirements
```
python 3.6
tensorflow 1.12
numpy
matplotlib
progressbar
pickle
```
## Step 1 - Getting the sources
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

## Step 2 - Preprocess Imagenet 
Set the location of your imagenet tarballs using $IMAGENET_HOME (change it to your favorite dir)
```
mkdir /tmp/Imagenet
export IMAGENET_HOME=/tmp/Imagenet
cp Cascaded_Inference/Imagenet_tools/* $IMAGENET_HOME 
cd $IMAGENET_HOME 
./preprocess_imagenet.sh
cd -
cd TF_Models/official/resnet
```
## Step 3 - Fine tuning the cascade (according to section 6.2 in the article)
First, evaluate the pre-trained model (change the -num_gpus according to your machine):
```
python3 imagenet_main.py --eval_only --is_cascaded --classifier_id 2 --data_format channels_first -rv 2 -rs 50 -md checkpoints/clf2 -pmcp checkpoints/resnet_imagenet_v2_fp32_20181001 --data_dir $IMAGENET_HOME/tf_records --num_gpus 4
```
Second, train and evaluate the classifiers $clf_0$ and $clf_1$
```
python3 imagenet_main.py --fine_tune --te 20 --is_cascaded --classifier_id 0 --data_format channels_first -rv 2 -rs 50 -md checkpoints/clf0 -pmcp checkpoints/resnet_imagenet_v2_fp32_20181001 --data_dir $IMAGENET_HOME/tf_records --num_gpus 4
python3 imagenet_main.py --fine_tune --te 20 --is_cascaded --classifier_id 1 --data_format channels_first -rv 2 -rs 50 -md checkpoints/clf1 -pmcp checkpoints/resnet_imagenet_v2_fp32_20181001 --data_dir $IMAGENET_HOME/tf_records --num_gpus 4
```
The three runs above will produce the softmax dump files inside the "output_softmax" directory. These dumps will be further analyzed for cascaded inference performance using various confidence thresholds.
## Step 4 - Check the validation accuracies of the 3 component DNNs
```
cd ../../../Cascaded_Inference
python analyze_imagenet_labels.py -dataset_path $IMAGENET_HOME -softmax_dir_to_validate_against ../TF_Models/official/resnet/output_softmax 
```
This should generate the following output to the terminal window:
```
Classifier 0:
 Train top-1:0.5208
 Train top-5:0.7486
 Test top-1:0.4668
 Test top-5:0.7022
Classifier 1:
 Train top-1:0.6925
 Train top-5:0.885
 Test top-1:0.6284
 Test top-5:0.8431
Classifier 2:
 Train top-1:0.9033
 Train top-5:0.9854
 Test top-1:0.7651
 Test top-5:0.9321
 ```
The above accuracies correspond to the reported accuracies in columns 2-4 of table 3 of our article.

## Step 5 - Find confidence thresholds 
Run the following program to determine confidence thresholds (which will be stored to log files) for 21 diffrent epsilon values.
```
mkdir Logs
mkdir Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top1
mkdir Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top5
python find_deltas_given_imagenet_cst_softmax_dumps.py -dataset_path $IMAGENET_HOME -softmax_path ../TF_Models/official/resnet/output_softmax -output_dir Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top1
python find_deltas_given_imagenet_cst_softmax_dumps.py -dataset_path $IMAGENET_HOME -softmax_path ../TF_Models/official/resnet/output_softmax -output_dir Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top5 --top5
```
## Step 6 - Evaluate the cascaded inference performance, produce plots!
The following code produces the figures 4.d and 4.e from our article. It is done by evaluating the softmax dumps of the cascade, with respect to the confidence threshold values that were found in the previous step.
```
mkdir output_log_processing
python gradual_log_driven.py -dataset_path $IMAGENET_HOME -softmax_path ../TF_Models/official/resnet/output_softmax
python gradual_log_driven.py -dataset_path $IMAGENET_HOME -softmax_path ../TF_Models/official/resnet/output_softmax --top5
```
At the end of the quick evaluation, check the "output_log_processing" directory where you will be able to find the figures.
## Step 7 - Evaluate softmax as a confidence measure
This code produces the figure 5.d from section 7.
```
python gradual_log_driven_reliable_map.py -dataset_path $IMAENET_HOME -softmax_path ../TF_Models/official/resnet/output_softmax
```
At the end of the run, the figure named 'Alpha_Accuracy_Confidence_Incidence_Bar.png' should be at "output_log_processing"
