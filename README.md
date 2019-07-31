# Sydney Innovation Challenge Submission

## Training Code

Please contact us separately for training code (dtia4818@uni.sydney.edu.au). The testing script provided can be run, using the weights existing in the same directory, along with the SampleSubmission.csv file. This has been built using CUDA9 and python3 on a Linux environment. 

## Credit

Our preprocessing script was influenced by Benjamin Graham's solution in the Diabetic Retinopathy Detection challenge (https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801). The training script was influenced by Chanhu's kernel (https://www.kaggle.com/chanhu/eye-inference-num-class-1-ver3). Images from the Diabetic Retinopathy Detection challenge were also used in addition the existing data found in the Sydney Innovation Challenge.

## Ensembling Code

This has not been provided as it was manually performed through a majority vote between 3 models in excel. The weights of the 3 models have been provided. The source code for the driver script contains the kappa-optimised threshold information for each of these models.

## Dependencies for Testing Code

OpenCV, pyTorch, scipy, numpy, pandas, PIL, efficientnet_pytorch, apex
