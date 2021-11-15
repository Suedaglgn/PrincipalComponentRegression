# Principal Component Regression (PCR)
PCR is a regression analysis technique that is based on principal component analysis (PCA).

PCR is a regressor composed of two steps: first, PCA is applied to the training data, possibly performing dimensionality reduction; then, a regressor (e.g. a linear regressor) is trained on the transformed samples. In PCA, the transformation is purely unsupervised, meaning that no information about the targets is used. As a result, PCR may perform poorly in some datasets where the target is strongly correlated with directions that have low variance. Indeed, the dimensionality reduction of PCA projects the data into a lower dimensional space where the variance of the projected data is greedily maximized along each axis. Despite them having the most predictive power on the target, the directions with a lower variance will be dropped, and the final regressor will not be able to leverage them. [Source](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)

## Principal Component Analysis (PCA)

Principal component analysis(PCA) is considered as one of the most popular technique for linearly independent feature extractiond and dimensionality reduction. We sometimes have machine learning problems in which input features have very high dimensions, which complicates machine learning, increasing processing and reducing accuracy. So, our first task is to reduce high dimensional input feature space, to a lower dimensional space which is more effective in machine learning task. So PCA has several benefits including data compression, improved visualization, increasing performance, simplifying machine learning models etc.

It is important to note that it is not only possible to reduce dimensionality of input feature space using PCA while retaining most of the variability of the data, but it is also possible to reconstruct the the original data through back projection techniques. [Source](https://www.kaggle.com/hassanamin/principal-component-analysis-with-code-examples)

### PCA Hyperparameters

- feature_dim: Input dimension. 
  - **Required.** 
  - _Valid values: positive integer_
- mini_batch_size: Number of rows in a mini-batch. 
  - **Required.** 
  - _Valid values: positive integer_
- num_components: The number of principal components to compute. 
  - **Required.**
  - _Valid values: positive integer_
- algorithm_mode: Mode for computing the principal components. 
  - **Optional.** 
  - _Valid values: regular or randomized._ 
  - _Default value: regular_
- extra_components: As the value increases, the solution becomes more accurate but the runtime and memory consumption increase linearly. The default, -1, means the maximum of 10 and num_components. Valid for randomized mode only. 
  - **Optional.** 
  - _Valid values: Non-negative integer or -1._ 
  - _Default value: -1._
- subtract_mean: Indicates whether the data should be unbiased both during training and at inference. 
  - **Optional.** 
  - _Valid values: One of true or false._
  - _Default value: true_

[Source](https://docs.aws.amazon.com/sagemaker/latest/dg/PCA-reference.html)

### Necessary Packages
 ```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
import matplotlib.pyplot as plt
```
## Dataset: Marketing Campaign

Context

A response model can provide a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses. The objective is to predict who will respond to an offer for a product or service.

Content

- AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
- Complain - 1 if customer complained in the last 2 years
- DtCustomer - date of customer’s enrolment with the company
- Education - customer’s level of education
- Marital - customer’s marital status
- Kidhome - number of small children in customer’s household
- Teenhome - number of teenagers in customer’s household
- Income - customer’s yearly household income
- MntFishProducts - amount spent on fish products in the last 2 years
- MntMeatProducts - amount spent on meat products in the last 2 years
- MntFruits - amount spent on fruits products in the last 2 years
- MntSweetProducts - amount spent on sweet products in the last 2 years
- MntWines - amount spent on wine products in the last 2 years
- MntGoldProds - amount spent on gold products in the last 2 years
- NumDealsPurchases - number of purchases made with discount
- NumCatalogPurchases - number of purchases made using catalogue
- NumStorePurchases - number of purchases made directly in stores
- NumWebPurchases - number of purchases made through company’s web site
- NumWebVisitsMonth - number of visits to company’s web site in the last month
- Recency - number of days since the last purchase

Acknowledgements

O. Parr-Rud. Business Analytics Using SAS Enterprise Guide and SAS Enterprise Miner. SAS Institute, 2014.

Inspiration

The main objective is to train a predictive model which allows the company to maximize the profit of the next marketing campaign.

[Source](https://www.kaggle.com/rodsaldanha/arketing-campaign)

## Further

Theoretical Knowledge: [PCA & PCR](https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/index)

## Example

 - Dataset: Marketing_Campaign.csv 
 - Preprocessing:
   - Data Information
   - Check for null values 
   - Encode Categorical Variables
 - Evaluation Metric: RMSE, R2
 - Validation: RMSE Cross Validation
 - Best Model: Determine optimum count of component
