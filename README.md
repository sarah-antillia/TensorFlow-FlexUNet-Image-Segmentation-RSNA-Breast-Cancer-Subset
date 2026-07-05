<h2>TensorFlow-FlexUNet-Image-Segmentation-RSNA-Breast-Cancer-Subset (2026/07/06)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>RSNA-Breast-Cancer-Subset</b> based on our 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and a 512x512 pixels PNG 
<a href="https://drive.google.com/file/d/1BUnBJJ8c4XgfaO663doMSFRhzReQed8G/view?usp=sharing">
<b>RSNA-Breast-Cancer-ImageMask-Subset.zip</b></a> 
(<a href="https://creativecommons.org/licenses/by-nc/4.0/legalcode.en">CC BY-NC 4.0)</a> with <b>AI generated pseudo colorized masks </b>(Benign:green, Malignant:red) 
which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs">
<b>RSNA Breast Cancer Detection - 512x512 pngs</b> </a> by Theo Viel.
<br><br>
<hr>
<b>Actual Image Segmentation for RSNA-Breast-Cancer-Subset Images of 512x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks,
except the first and third cases.
<br><br>
<b>class_color_map = {Benign:green, Malignant:red}</b><br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/3851_1995146754.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/3851_1995146754.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/3851_1995146754.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/37598_1981738993.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/37598_1981738993.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/37598_1981738993.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/37358_387709760.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/37358_387709760.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/37358_387709760.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/orvile/aisslab-breast-cancer-dataset">
<a href="https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs">
<b>RSNA Breast Cancer Detection - 512x512 pngs</b> </a><br>
<b>Processed dicoms & resized to 512x512 for easier use</b> by Theo Viel.
<br><br>
The following explanation was taken from the above web site.
<br><br>
<b>About Dataset</b><br>
Generated using <a href="https://www.kaggle.com/code/theoviel/dicom-resized-png-jpg">
https://www.kaggle.com/code/theoviel/dicom-resized-png-jpg
</a>
<br>
<ul>
<li>256x256 : <a href="https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-256-pngs">
https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-256-pngs
</a></li>
<li>512x512 : <a href="https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs">
https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-512-pngs
</a></li>
<li>1024x1204:<a href=" https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-1024-pngs">
https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-1024-pngs
</a></li>
</ul>
The following description was taken from <a href="https://github.com/RSNA/AI-Challenge-Data/wiki/RSNA-Screening-Mammography-Breast-Cancer-Detection">
RSNA Screening Mammography Breast Cancer Detection
</a>.<br>
<br>
RSNA assembled this dataset in 2022 for the RSNA Screening Mammography Breast Cancer Detection AI Challenge 
(<a href="https://www.kaggle.com/competitions/rsna-breast-cancer-detection/">
https://www.kaggle.com/competitions/rsna-breast-cancer-detection/</a>). RSNA collected de-identified screening mammograms and supporting information from two sites, totaling just under 20,000 imaging studies.
 </a>). RSNA collected de-identified screening mammograms and supporting information from two sites, 
 totaling just under 20,000 imaging studies.
<br><br>
<b>Download</b><br>
<a href="https://mira.rsna.org/dataset/3">Medical Imaging Resource for AI (MIRA)</a>
<br><br>
<b>License</b><br>
You may access and use these de-identified imaging datasets and annotations (“the data”) for 
non-commercial purposes only, including academic research and education, 
as long as you agree to abide by the following provisions: 
Not to make any attempt to identify or contact any individual(s) who may be the subjects of the data.
<br>
<br>
<h3>
2 RSNA-Breast-Cancer-Subset ImageMask Subset
</h3>
<h3>
2.1 Download RSNA-Breast-Cancer-Subset ImageMask Subset
</h3>
 If you would like to train this RSNA-Breast-Cancer-Subset Segmentation model by yourself,
please download our dataset 
<a href="https://drive.google.com/file/d/1BUnBJJ8c4XgfaO663doMSFRhzReQed8G/view?usp=sharing">
<b>RSNA-Breast-Cancer-ImageMask-Subset.zip</b></a> 
(<a href="https://creativecommons.org/licenses/by-nc/4.0/legalcode.en">CC BY-NC 4.0)</a>
on the google drive,
expand the downloaded , and put it under <b>./dataset/</b> to be.
<pre>
./dataset
└─RSNA-Breast-Cancer-Subset
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>RSNA-Breast-Cancer-Subset Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/RSNA-Breast-Cancer-Subset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br><br>
<h3>
2.2 Derivation of RSNA-Breast-Cancer-ImageMask-Subset
</h3>
The folder structure of the original <b>RSNA-Breast-Cancer-Detection</b> is the following. It contains no annotation (mask) files,
because it is an image classification dataset,
<pre>
./RSNA_Breast_Cancer_Detection (54,707 PNG files)
├─5_640805896.png
├─5_940388076.png
...
└─65534_1888933323.png
   
</pre>
<b>Step 1</b><br>
We generated our own <b>pseudo colorized masks </b> (Benign:green, Malignant:red) corresponding to all 54,707 PNG files in RSNA_Breast_Cancer_Detection folder  
by applying an inference (segmentation) method of
a pretrained FlexUNet model <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-AISSLab-Breast-Cancer">
TensorFlow-FlexUNet-Image-Segmentation-AISSLab-Breast-Cancer
</a> to all RSNA_Breast_Cancer images, without human annotation experts.
<br><br>
<b>Step 2</b><br>
We generated our own RSNA-Breast-Cancer-ImageMask-Dataset from all pairs of RSNA_Breast_Cancer images and
their corresponding pseudo masks by excluding all empty black masks and their corresponding images. 
<br>
<br>
<b>Step 3</b><br>
We finally generated <b>RSNA-Breast-Cancer-ImageMask-Subset</b> from RSNA-Breast-Cancer-ImageMask-Dataset  
by reducing the fullset dataset to one-fifth of its total size, because the number of image and mask 
files of the fullset is 40,293 respectively, which is too large to use for our experiment.<br>
<br>
<b>
You could potentially create a more refined pseudo masks from RSNA-Breast-Cancer images
by applying a segmentation method of
a pretrained FlexUNet model trained by <b>RSNA-Breast-Cancer-ImageMask-Subset</b>
</a> to all cancer images.
</b>
<br>
<br>
<h3>
2.3 Train Sample Images and Masks
</h3>
<b>Train Sample Images</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train Sample Masks</b><br>
As shown below, some images contain a mixture of Benign(green) and Malignant(red) areas.<br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained RSNA-Breast-Cancer-Subset TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 3
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for RSNA-Breast-Cancer-Subset 3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;RSNA-Breast-Cancer-Subset 1+2
;                  Benign:green, Malignant:red
rgb_map={(0,0,0):0,(0,255,0):1,(255,0,0):2,}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = False
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br>
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved.
 
 <br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (10,11,12)</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (21,22,23)</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stoppd at epoch 23 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/train_console_output_at_epoch23.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for RSNA-Breast-Cancer-Subset.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/evaluate_console_output_at_epoch23.png" width="880" height="auto">
<br><br>Image-Segmentation-RSNA-Breast-Cancer-Subset

<a href="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this RSNA-Breast-Cancer-Subset/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0207
dice_coef_multiclass,0.9894
</pre>
<br>
<h3>5 Inference</h3>
Please move to  <b>./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for RSNA-Breast-Cancer-Subset.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  RSNA-Breast-Cancer-Subset  Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, except the second and third cases.
<br>
<br>
<b>class_color_map = {Benign:green, Malignant:red}</b><br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/9705_42089822.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/9705_42089822.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/9705_42089822.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/15136_639145075.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/15136_639145075.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/15136_639145075.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/5365_633905254.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/5365_633905254.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/5365_633905254.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/3324_694378439.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/3324_694378439.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/3324_694378439.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/10566_1932026673.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/10566_1932026673.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/10566_1932026673.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/images/44454_1022268346.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test/masks/44454_1022268346.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/RSNA-Breast-Cancer-Subset/mini_test_output/44454_1022268346.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-AISSLab-Breast-Cancer</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-AISSLab-Breast-Cancer">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-AISSLab-Breast-Cancer
</a>
<br><br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-ISPY1-Breast-Cancer</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-ISPY1-Breast-Cancer">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-ISPY1-Breast-Cancer
</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-Mammogram-Mastery</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-Mammogram-Mastery">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Breast-Cancer-Mammogram-Mastery
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-MIAS-Mammogram
</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
