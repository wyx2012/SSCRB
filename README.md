#Type: Package

Files: 1.data

circRNA-RBP:37 datasets

2.code

pairs.py 

wTest.py

wModle.py

wyxPreData.py

WOwnDataset.py


The tool is developed for circRNA-RBP interaction sites identification using deep hierarchical network
# Requirements
python                    3.8.10
numpy                     1.22.4   
pandas                    2.0.3 
scikit-learn              1.3.0    
torch                     1.11.0+cu113 
torchvision               0.12.0+cu113  
# Usage

1.Firstly, fill in the circRNA name you need to predict in pairs.py, run pairs.py, and generate the corresponding circRNA structure file in the Dataset folder.

2. Fill in the circRNA name you need to predict in wTest.py, run wTest.py, and the results will be saved in the RESULT file of the corresponding circRNA in the dataset.

Thank you and enjoy the tool!# MMD-DTI
