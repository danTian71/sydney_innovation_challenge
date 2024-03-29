# Sydney Innovation Challenge Submission

## Test Code Usage

The weights can be found at https://drive.google.com/drive/folders/1jfbeZTejgcIwyweGI8R6O4TXJZLs2cOf?usp=sharing. The testing script requires the weights in the same directory, but otherwise can be run if the dependencies are installed and the SampleSubmission.csv file is in the same directory. The weights were not uploaded on to github due to their size.

Run it through the command: python3 DRIVER.py path/to/test/images/ output_name.csv

## Training Code

Please contact us at dtia4818@uni.sydney.edu.au

## Credit

Our preprocessing script was influenced by Benjamin Graham's solution in the Diabetic Retinopathy Detection challenge (https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801). The training script was influenced by Chanhu's kernel (https://www.kaggle.com/chanhu/eye-inference-num-class-1-ver3). Images from the Diabetic Retinopathy Detection challenge were also used in addition the existing data found in the Sydney Innovation Challenge.

## Ensembling Code

This has not been provided as it was manually performed through a majority vote between 3 models in excel. The weights of the 3 models have been provided. The source code for the driver script contains the kappa-optimised threshold information for each of these models.

## Dependencies for Testing Code

OpenCV, pyTorch, scipy, numpy, pandas, PIL, efficientnet_pytorch, apex
