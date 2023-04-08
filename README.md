# Flipkart Laptop Price Prediction

## Flipkart Laptop Data - Business Insights on Product Pricing

Flipkart is an Indian e-commerce company that was founded in 2007 by Sachin Bansal and Binny Bansal (not related). The company is headquartered in Bengaluru, India, and initially started as an online bookstore before expanding to other categories such as electronics, fashion, and home goods.

In 2018, Walmart acquired a 77% stake in Flipkart for $16 billion, making it the largest ever acquisition by Walmart. As of 2021, Flipkart is one of the leading e-commerce companies in India and competes with other major players such as Amazon and Reliance JioMart.


## Problem Statement

Elon Musk has started his own laptop company (Tesla Laptop) in India. He wants to give tough fights to big companies like Apple, Alienware, HP etc.

<p align="center">
  <img  src="https://github.com/Abdul-Jaweed/Laptop-Price-Prediction/blob/main/Images/elon.PNG">
</p>

He does not know how to estimate the price of laptops his company creates. In this competitive laptop market you cannot simply assume things. 

Musk wants to find out some relation between features of a Laptop(eg:- RAM, OS,  etc) and its selling price. But he is not so good at Data Mining. 😥

To solve this problem he already asked his Data Engineering team (from Twitter 🐥🤭) to collect laptop data from various competitors.


<p align="center">
  <img  src="https://github.com/Abdul-Jaweed/Laptop-Price-Prediction/blob/main/Images/elon1.PNG">
</p>



Now he needs your (i.e Data Scientist’s) help to solve the following problem. 🥰



# Sprint - 1 (Data Analysis and Mining)

## Musk’s Requirements

- Build an interface (using streamlit) for Elon Musk where he can enter the laptop features like RAM Size, RAM Type, HDD, OS, etc. and get the price prediction.

- Based on your Data Analysis and MIning skills, give recommendations to Elon Musk on how the pricing works in the laptop market.


Note - This is real world data. It was scrapped from Flipkart on 21-12-2022 at 11:50 AM. 
Kindly refer below mentioned notebook to see how the data was scraped by Musk’s Data Engineering Team:

[Click here to get the code for web scraping and cleaning the Flipkart data](https://github.com/bansalkanav/Machine_Learning_and_Deep_Learning/tree/master/Module%206%20-%20Case%20Studies/8.%20Regex%20and%20Webscrapping/Web%20Scrapping)


# Sprint - 2 (ML Model Tuning)

Congratulations! Elon Musk is happy with your models and insights that you have shared in the previous requirements. 

Now he wants you to test the models you have created for Underfitting and Overfitting. He wants you to build a best fit model.

## Todo - 

 - First try to check if the models you have created are a best fit or not.
 - See if applying cross validation helps improve your models performance.


 # Dataset [link](Dataset\laptop_details.csv)

## Web Scrapping Data from Flipkart Credit goes to [KANAV BANSAL](https://www.linkedin.com/in/kanavbansal/) SIR 😊

### EDA, Feature Engineering and Model Building Credit goes to me [ABDUL JAWEED](https://www.linkedin.com/in/abdul-jaweed-datascientist/) 😅

**Flipkart Laptop Price Prediction Jupyter Notebook**

<p align="center">
  <img  src="Images\0.PNG">
</p>



**Initial Observations of the Dataset**
 - MRP column is a Target variable
 - Feature column is required a lot of Feature Engineering 
 - Product column is also required a little bit of Feature Engineering 
 - For our requirement Rating column won't be needed


# Exploratory Data Analysis

Exploratory data analysis (EDA) is used by data scientists to analyze and investigate datasets and summarize their main characteristics, often employing data visualization methods.


**Findings**

- Data have 720 Rows and 4 Columns
- MRP need treatment because it contain strings 
- Likewise we dont use Rating Columns so we dont need to take care of that column
- We have limited amount of data so we don't to need drop the data in duplicated values or any others



# Data Preprocessing

Data preprocessing is the process of transforming raw data into a format that can be easily understood and analyzed by a computer. This involves cleaning and transforming the data, integrating data from different sources, reducing the dataset size by eliminating irrelevant data, and discretizing continuous data. Data preprocessing is essential to ensure that the data is accurate, consistent, and in a format suitable for analysis.


**Findings**

- Minimum Values of MRP Column is 14990
- Maximum Values of MRP Column is 434830
- Mean Values of MRP Column is 81605
- Median Values of MRP Column is 59990
- 25% Values of MRP Column is 38996
- 75% Values of MRP Column is 95240

#### Which means data have to many outlier


**Visualizing Density PLot in MRP Column for checking the Distribution**

<p align="center">
  <img  src="Images\1.png">
</p>

**Visualizing BoxPLot in MRP Column for checking the Outlier**

<p align="center">
  <img  src="Images\2.png">
</p>


### After checking the distribution of the MRP column
**Findings**

- Like i previously mension we have too many Outliers
- Instead of removing values we have to use Data transformation to handle Outliers


### Data transformation

Data transformation is the process of converting data from one format or scale to another to make it suitable for analysis. It can involve changing the distribution of data, scaling or normalizing values, converting categorical data to numerical data, and more.


**Log transformation** converting data to a logarithmic scale to reduce the magnitude of extreme values and make the distribution more symmetric.

<p align="center">
  <img  src="Images\3.png">
</p>


<p align="center">
  <img  src="Images\4.png">
</p>



**Reciprocal** transformation involves taking the reciprocal of each value in a dataset to reduce the effect of extreme values and make the distribution of the data more symmetric. It is particularly useful when the data has a positive skew, but has limitations and should be used with caution.

<p align="center">
  <img  src="Images\5.png">
</p>


<p align="center">
  <img  src="Images\6.png">
</p>

**Square root transformation** involves taking the square root of each value in a dataset to reduce the effect of extreme values and make the distribution of the data more symmetric. It is often used when the data has a positive skew, but its effectiveness depends on the specific dataset and analysis goals.

<p align="center">
  <img  src="Images\7.png">
</p>


<p align="center">
  <img  src="Images\8.png">
</p>

**Power transformation** - raising data to a power to adjust the distribution and reduce the impact of outliers.


<p align="center">
  <img  src="Images\9.png">
</p>

<p align="center">
  <img  src="Images\10.png">
</p>


**Findings**

- After applying few data transformation log and power will perform better
- I did some parameter tuning in power transformation to make it better
- Conclusion i choose Log Transformation for my MRP Column
