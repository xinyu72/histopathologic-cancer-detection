# Histopathologic Cancer Detection

**Author: Boyang Wei, Xinyu Zhang**

This project focus on dentifying metastatic tissue in histopathologic scans of lymph node sections by Image Processing and Convolutional Neural Networks. Particularly, we explored how **image augmentation** methods help optimize the CNN model. 
. 
There are three main sections:

* Image Preprocessing [Image_Prepocessing.py](Image_Prepocessing.py)
	* Removing outliers (defective images)
	* Balance the sample (training and testing)
* Image Augmentation [Augmentation.py](Augmentation.py)
	* Blur + Sharping Combination (linear filter)
	* Gaussian Blur 
	* White/Contrast
* Convolutional Neural Networks [CNN.py](CNN.py)
	* Building layers and parameters
	* Training for 3 different datasets 
	* Model Evaluation (loss/accuracy/confusion matrix/AUC)

The entire pipeline can also be foudned in [Image_Augmentation_CNN_Exploration.ipynb](Image_Augmentation_CNN_Exploration.ipynb), which includes code and entire output from all sections above.

## Dataset and Background

The data are coming from [PatchCamelyon (PCam) benchmark](https://www.kaggle.com/c/histopathologic-cancer-detection/data) (kaggle competition)

Particularly, the entire dataset consists of 278,000 scans of lymph node sections with labels (cancer/non-cancer).
Each picture has size of 96 x 96 pixels with monochromatic pattern. There are 220,000 (training) + 57,000 (testing) samples in total.
Entire dataset are 60 / 40 split of negative to positive samples.


## Goal and Hypothesis

In this project, our goal is to create an algorithm to identify metastatic cancer in small image patches taken from larger digital 
pathology scans in term of binary image classification.

Particularly, we want to test if image augmentation plays important role in influencing the model performance.

## Result and Conclusion

Among all three different preprocessed dataset,  data augmented with linear filter tend to have best test accuracy given the same CNN model.

These results confirmed our hypothesis that augmentation not only helps reduce overfitting chance, but also optimizes the training/testing error to some extends. 

## Additional Insights

Choosing the right augmentation methods/filters depend on the types of images and questions we have. 

There’s no universal answer to solve every problem and parameters for filtering and training the model. 

Domain knowledge, combined with trial and error are important for optimizing the results with contents. 

The process is still an ongoing research and we will investigate more when time permitted.





