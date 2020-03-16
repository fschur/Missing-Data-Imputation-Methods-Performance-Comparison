# Missing-Data-Imputation-Methods-Performance-Comparison
The data imputation methods [MissForest](https://cran.r-project.org/web/packages/missForest/missForest.pdf), [GAIN](https://arxiv.org/abs/1806.02920), [MICE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/), MICE-NN and [MIWAE](http://proceedings.mlr.press/v97/mattei19a.html) are tested on two UCI datasets (Dataset for Sensorless Drive Diagnosis Data Set, Page Blocks Classification Dataset). MICE-NN is a modified version of MICE, where instead of linear regresssion fully connected neural networks are used. The tests are done by taking the complete dataset (without missing values) introducing either MAR or MCAR missingness with the desired missing rate and then using the imputation methods to impute the missing values. Since the correct values are known, the real MSE can be computed. To test other datasets, save the dataset as a 2-dim numpy vector in the folder [data](data). Now set dataset = "name" when calling the imputation method, where your dataset in the folder [data](data) is named "name_y" and name_x.

MCAR missing values are introduced by dropping each value in the data independently with probability "p_miss". MAR missing values are introduced by summing over one third of each observation and dropping each value in the rest of the observation independently with a probability proportional to the computed sum. For this the variable "para" is used (for details see load_data in utils.py).



## Requirements
The code requires Python 3.6 or later.
Required packages are:

* fanyimpute >= 0.5.3
* mathplotlib >= 2.2.2
* missingpy >= 0.2.0
* numpy >= 1.16.2
* pathlib >= 2.3.3
* pickle 
* Pillow >= 5.4.1
* pylab 
* scipy >= 1.2.1
* sklearn 
* tensorflow >= 1.14
* tensorflow_probability >=0.7.0
* torch >= 1.0.1
* torchvision >= 0.2.2
* tqdm >= 4.31.1

