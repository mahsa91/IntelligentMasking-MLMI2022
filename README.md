Intelligent Masking: Deep Q-Learning for Context Encoding in Medical Image Analysis
====
Here is the code of our paper named ["Intelligent Masking: Deep Q-Learning for Context Encoding in Medical Image Analysis"](https://arxiv.org/pdf/2203.13865.pdf)" accepted in MLMI 2022. 

The need for a large amount of labeled data in the supervised setting has led recent studies to utilize self-supervised learning to pretrain deep neural networks using unlabeled data. Intelligent masking is a novel self-supervised approach that occludes targeted regions to improve the pre-training procedure. It is designed based on a reinforcement learning-based agent which learns to intelligently mask input images through deep Q-learning. We show that training the agent against
the prediction model can significantly improve the semantic features extracted for downstream classification tasks.

The code in this repository has two main parts: 
- Pre-training a model with our proposed approach (intelligent_masking.ipynb)
- Evaluating the performance of a classifier with the pre-trained encoder (from previous step) in its back-bone

<a href="url"><img src="https://github.com/mahsa91/IntelligentMasking-MLMI2022/blob/main/intelligent-masking.JPG" align="center" height="350" ></a>  

Usage 
------------
The main file for pre-training step is "intelligent_masking.ipynb" and the main file for classification task is "classification.ipynb".


Input Data
------------
To run our pre-training model, you can see "intelligent_masking.ipynb". To run this notebook for your data, you need to write a function to get your dataset. The output of your function should be a dataframe containing two columns. The firs one  is 'image_path' which contains the path of images, and the second one is 'label_cat' which contains the categorical labels. Then after splitting to train and test, two dataloaders from training and testing data are created. In "Model Config", the structure of masking network (agent) and reconstruction network are defined. In "Trainer Config", the parameters for training algorithm are set. Then the "model" variable is the trained model including reconstruction component (trained encoder and decoder) and masking agent. The parameters for saving model is also defined in "checkpoint_callback" variable. 

To run classification step, you can see "classification.ipynb. To run this notebook, you should define the settings in setup cell.

Metrics
------------
Accuracy and F1 score and AUROC are calculated in the code. 


Bug Report
------------
If you find a bug, please send email to bahrami.mojtaba.93@gmail.com or mahsa.ghorbani@sharif.edu. Please attach including your input file and the parameters that caused the bug (if necessary).
We would appreciate if you send us any comment or suggestion about the model.
