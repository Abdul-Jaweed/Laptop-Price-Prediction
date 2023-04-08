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



**Outlier**

An outlier is an observation or data point that is significantly different from other observations in a dataset. In other words, an outlier is a value that is either much larger or much smaller than the other values in the dataset.

Outliers can occur for various reasons, such as measurement errors, data entry mistakes, or they may represent true extreme values in the population. However, outliers can have a significant impact on statistical analyses, as they can distort the results, affecting measures of central tendency, variability, and correlation.

Therefore, it is important to identify and handle outliers appropriately in data analysis. Some common techniques for handling outliers include capping or trimming outliers, imputing missing values, or using robust statistical methods that are less sensitive to outliers. Ultimately, the choice of technique will depend on the nature of the data and the objectives of the analysis.



**Capping Outlier**

Capping outliers means setting a limit to extreme values in a dataset. It helps to reduce the impact of outliers on the analysis and make the dataset more representative of the general population. However, caution should be exercised as it may mask important information in the data and may not be appropriate in all situations. Various methods are used to cap outliers, depending on the data and objectives of the analysis.


<p align="center">
  <img  src="Images\11.png">
</p>

### Trimming

Trimming outliers means removing extreme values from a dataset by setting a threshold value and deleting any values that exceed this limit. It helps to remove the impact of outliers on the analysis and make the dataset more representative of the general population. However, caution should be exercised as it may also mask important information in the data and may not be appropriate in all situations. Various methods are used to trim outliers, depending on the data and objectives of the analysis.

<p align="center">
  <img  src="Images\12.png">
</p>

**Finding**

Since Trying both Capping and Trimming Technique but didn't effect on distribtion so I decided so use capping instead trimming because of less data.



# Feature Engineering

Feature engineering is the process of transforming raw data into useful features that can be used to train machine learning models. It involves selecting and extracting relevant data features, as well as transforming and scaling them to improve model performance. Feature engineering requires domain knowledge, creativity, and analytical skills and involves techniques such as one-hot encoding, scaling, feature selection, and feature extraction. The goal is to identify and select relevant features while avoiding overfitting the model to the training data. It is a critical component of the machine learning pipeline to develop accurate and robust predictive models.



**Findings**

- In Product Column we can fetch Brand Name and Ram Size 
- But in Feature we can fetch more Data regarding our project
 
 **My Observation I'm thinking to Extract all this data mention below**

 1. -  Brand Name  
 2. -  RAM Type    
 3. -  RAM Size    
 4. -  OS
 5. -  Disk Type   
 6. -  Disk Size   
 7. - Processor
 8. -  GPU 
 9. -  Warranty
 10. - Screen Size



 **Brand Name**

A laptop brand name refers to the name of the company that manufactures and sells laptops. There are numerous laptop brands in the market, each with its unique features, design, and price range. Popular laptop brands include Apple, Dell, HP, Lenovo, Acer, ASUS, MSI, Razer, Samsung, and LG. Each of these brands has its strengths and weaknesses, and consumers often choose a laptop brand based on their specific needs and preferences.

**Finding**

- To prevent outlier less brand values store into Others Variable



**Operating System (OS)**

An operating system (OS) is a software that manages a computer's hardware and provides services for running applications. It serves as a bridge between the hardware and the software, enabling users to interact with the computer and its resources. Examples of popular operating systems for laptops include Windows, macOS, and Linux. Each operating system has its unique interface, features, and compatibility with applications. The operating system manages tasks such as memory allocation, process management, file systems, and security. It also provides a platform for developers to create and run applications that work seamlessly with the hardware and other software components. The choice of operating system can significantly impact a laptop's performance, usability, and security, and it's essential to choose the one that best fits your needs and preferences.


**Finding**

- Windows Laptops are more in compare to others 
- Apple Laptops is less because of expensive
- Others maybe linux or andriod OS



**RAM Type**

Laptop RAM (Random Access Memory) type refers to the type of memory technology used in a laptop to temporarily store data that the CPU (Central Processing Unit) needs to access quickly. There are several types of laptop RAM available in the market, including DDR (Double Data Rate) SDRAM (Synchronous Dynamic Random Access Memory), DDR2 SDRAM, DDR3 SDRAM, DDR4 SDRAM, and DDR5 SDRAM. Each type of RAM offers different levels of speed, power consumption, and capacity. DDR4 SDRAM is currently the most commonly used type of RAM in laptops, offering a good balance of speed, power efficiency, and capacity. When buying a laptop, it's essential to consider the type and amount of RAM the laptop has, as it can significantly impact its performance and multitasking capabilities.


**Findings**

- To prevent overfitting less ram type values store into Others Variable




**RAM**

Laptop RAM (Random Access Memory) is a type of computer memory that stores data temporarily and allows the CPU (Central Processing Unit) to access it quickly. RAM plays a crucial role in the performance of a laptop, as it helps to determine how many applications and processes the laptop can handle simultaneously. The amount of RAM in a laptop can range from 2GB to 32GB or more, and it can be upgraded in some cases. The speed of the RAM, measured in MHz or GHz, also affects the laptop's performance. Faster RAM allows for quicker data transfer and improves overall system responsiveness. When purchasing a laptop, it's important to consider the amount and speed of the RAM to ensure optimal performance for your specific needs.


**Findings**

- 8 GB RAM Laptops are more in compare to others. 
- Two laptops RAM size is 128 GB. It will affect the ML Model so I decided so merge into 32 GB RAM size.


**Disk Type**

Disk type refers to the physical storage device used in a laptop or other computing device to store data, programs, and other digital content. There are two main types of disk storage: hard disk drives (HDD) and solid-state drives (SSD).

HDDs are the traditional type of disk storage, and they consist of spinning disks that store data on magnetic surfaces. HDDs are generally cheaper and offer larger storage capacities than SSDs, but they are also slower and more prone to failure due to their mechanical components.

SSDs are a newer type of disk storage that uses flash memory to store data. SSDs are faster and more reliable than HDDs, and they consume less power, which can result in longer battery life for laptops. However, SSDs are generally more expensive than HDDs, and they offer smaller storage capacities.

In recent years, hybrid drives have also become available, which combine the large storage capacity of an HDD with the speed and reliability of an SSD. Hybrid drives use a smaller amount of solid-state memory as a cache to speed up frequently accessed data, while storing less frequently accessed data on the traditional spinning disk.

When choosing a laptop, the type of disk storage is an important factor to consider, as it can affect the speed, performance, and cost of the laptop. If the user needs large amounts of storage at a lower cost, an HDD may be a better choice, while if speed and reliability are more important, an SSD may be a better choice.



**Disk Size**

Disk size refers to the amount of storage space available on a laptop's hard drive or solid-state drive (SSD). The disk size is typically measured in gigabytes (GB) or terabytes (TB).

The disk size is an important consideration when choosing a laptop, as it determines how much data the laptop can store. The amount of storage required will depend on the user's needs, such as whether they will be storing large files, such as photos or videos, or mainly using cloud storage.

Laptop disk sizes can vary widely, with smaller laptops typically offering 128GB or 256GB of storage, while larger laptops may offer 1TB or more. In recent years, it has become increasingly common for laptops to offer both a solid-state drive (SSD) and a traditional hard drive (HDD), with the SSD used for the operating system and frequently accessed files, while the HDD is used for storing larger files.

It is important to consider the amount of storage required when choosing a laptop, as it can affect the performance and usability of the device. A laptop with insufficient storage space may become slow and sluggish, while a laptop with too much storage space may be more expensive than necessary.


**Findings**

- 512 GB Storage Size are more in compare to others.
- To prevent Outlier I merge extreme values to thers values.


**Processor**

A laptop processor is the main component of a computer that carries out instructions and performs calculations necessary for various tasks. The processor is also known as the Central Processing Unit (CPU), and it is responsible for executing programs and running the operating system.

There are two major manufacturers of laptop processors, Intel and AMD. Intel's processors are generally considered to be the industry standard and are used in many laptops. AMD's processors are less common but are becoming increasingly popular.

Processors are characterized by several key features, including clock speed (measured in GHz), number of cores (which affects multitasking performance), cache size (which affects data transfer speeds), and power consumption. The higher the clock speed and number of cores, the more powerful the processor, but this also tends to result in higher power consumption and heat output.

In recent years, laptop processors have become more efficient and powerful, making it possible to perform demanding tasks such as gaming and video editing on a laptop. However, desktop processors still tend to be more powerful due to their larger size and better cooling capabilities.


**Findings**

- Majorly Two categories of Processor so i decided to categorize then. 


**Graphical Processing Unit (GPU)**

A laptop GPU (Graphics Processing Unit) is a specialized processor that is responsible for rendering graphics and images on a laptop. The GPU is designed to work in conjunction with the CPU to provide the necessary processing power for various visual applications.

There are two main manufacturers of laptop GPUs, NVIDIA and AMD. NVIDIA is the industry leader and its GPUs are used in a wide range of laptops, while AMD's GPUs are less common but are becoming more popular.

The performance of a laptop GPU is measured in terms of its clock speed, memory size, memory speed, and number of CUDA cores or stream processors. The clock speed is the frequency at which the GPU operates, while the memory size and speed determine the amount and speed of data that can be processed by the GPU. The number of CUDA cores or stream processors determines the number of parallel processing units within the GPU.

A powerful laptop GPU is essential for tasks such as gaming, video editing, and 3D modeling, as these applications require a lot of graphical processing power. However, a powerful GPU also tends to consume more power and generate more heat, which can affect battery life and laptop performance.

In recent years, laptop GPUs have become more efficient and powerful, making it possible to perform demanding visual tasks on a laptop. However, for the most demanding tasks, a desktop GPU still tends to be more powerful due to its larger size and better cooling capabilities.


**Findings**

- Majorly Laptops and Integrated Graphics. 
- AMD GPU is more as compare to others.
- NVIDIA GPU is less just because of expensive.
- Intel are play majorly Integrated Graphics.



**Warranty**

A laptop warranty is a guarantee that the manufacturer provides to cover defects in materials and workmanship for a specific period of time. The warranty period varies by manufacturer, but it typically ranges from one to three years.

The warranty covers the cost of repairing or replacing parts that fail due to defects in materials or workmanship. However, it does not cover damage caused by accidents, misuse, or normal wear and tear. Some warranties may also exclude certain types of components, such as batteries or accessories.

To make a claim under the warranty, the owner of the laptop must contact the manufacturer or its authorized service provider. The manufacturer will then diagnose the problem and either repair or replace the defective part.

It is important to note that warranties may vary in terms of what they cover and the extent of the coverage. Some warranties may include additional services, such as technical support or on-site repairs, while others may have limitations or exclusions.

It is recommended to read the terms and conditions of the warranty carefully before purchasing a laptop, and to consider purchasing an extended warranty if one is available, especially if the laptop is used for critical work or if it is an expensive model.



**Findings**

- Majorly Laptops comes 1 Year of warranty
- 3 Year Warranty Laptops are only 8 so I decided to marge into 2 Year Warranty Laptops just because of preventy Outlier.



**Screen Size**

Laptop screen size refers to the diagonal measurement of the display screen, usually measured in inches. Screen size is an important factor to consider when choosing a laptop, as it can affect the user's comfort and the laptop's portability.

Laptop screen sizes typically range from 11 inches to 17 inches, with the most common sizes being 13 inches, 14 inches, and 15 inches. Smaller screens are generally more portable and lightweight, while larger screens offer more screen real estate and are better suited for tasks such as video editing or gaming.

Screen resolution is another important factor to consider, as it affects the clarity and sharpness of the displayed images. Higher resolutions result in sharper and more detailed images, but they also tend to be more demanding on the laptop's processing power and battery life.

In recent years, laptop screens have become more advanced and innovative, with features such as touchscreens, high refresh rates, and HDR support. These features can enhance the user's experience and productivity, but they also tend to increase the cost of the laptop.

When choosing a laptop, it is important to consider the user's needs and preferences, as well as their budget. A larger screen may be more comfortable for some users, while others may prioritize portability and opt for a smaller screen.


**Findings**

- 15.6 Inches laptops are more in compare to others
- In Screensize Column are they too many extreme value so I decided to cap into 14.0 and 15.6 Inches


**Noted :** 

Price Column are applied by Log Transformation



# Data Visualization 

Data Visualization is the graphical representation of data and information, which is an important part of data analysis. It allows analysts and stakeholders to quickly understand complex data and identify patterns and trends. There are many different types of data visualizations, including bar charts, line charts, scatter plots, heatmaps, pie charts, box plots, histograms, area charts, tree maps, bubble charts, and network diagrams. To create data visualizations, you can use various tools and libraries, such as Matplotlib, Seaborn, Plotly, and Tableau. When creating data visualizations, it is important to choose the appropriate chart type based on the type of data and the message you want to convey, and to use appropriate labels, titles, and colors to ensure the visualization is clear and easy to understand.



## Univariate Analysis

Univariate analysis is a statistical technique used to examine and summarize a single variable at a time. It involves analyzing the distribution, central tendency, and variability of the variable through measures such as frequency distribution, measures of central tendency, measures of dispersion, histograms, and box plots. Univariate analysis is an essential step in data analysis, providing a basic understanding of the data before proceeding to more complex multivariate analyses.


**Brand Name**


<p align="center">
  <img  src="Images\11.png">
</p>

<p align="center">
  <img  src="Images\12.png">
</p>


**Operating System (OS)**

<p align="center">
  <img  src="Images\13.png">
</p>

<p align="center">
  <img  src="Images\14.png">
</p>

**Ram Type**

<p align="center">
  <img  src="Images\15.png">
</p>

<p align="center">
  <img  src="Images\16.png">
</p>

**RAM**

<p align="center">
  <img  src="Images\17.png">
</p>

<p align="center">
  <img  src="Images\18.png">
</p>

**Processor**

<p align="center">
  <img  src="Images\19.png">
</p>

<p align="center">
  <img  src="Images\20.png">
</p>

**Graphics Processing Unit (GPU)**

<p align="center">
  <img  src="Images\21.png">
</p>

<p align="center">
  <img  src="Images\22.png">
</p>

**Warranty**

<p align="center">
  <img  src="Images\23.png">
</p>

<p align="center">
  <img  src="Images\24.png">
</p>

**Screen Size**

<p align="center">
  <img  src="Images\25.png">
</p>

<p align="center">
  <img  src="Images\26.png">
</p>

**Disk Type**

<p align="center">
  <img  src="Images\27.png">
</p>

<p align="center">
  <img  src="Images\28.png">
</p>

**DISK SIZE**

<p align="center">
  <img  src="Images\29.png">
</p>

<p align="center">
  <img  src="Images\30.png">
</p>

**Price**

<p align="center">
  <img  src="Images\31.png">
</p>

<p align="center">
  <img  src="Images\32.png">
</p>


<p align="center">
  <img  src="Images\33.png">
</p>


## Findings


**Brand Name**

- Asus lead the laptop market and the second lead is Lenovo.
- The Market Share of Asus is 35% and lenovo is 37%.
- Dell, HP and Acer are holding 22% of Market share.
- Some Laptop Market share is less because its only target premium gaming market.
- Apple Market share is 3%. Maybe of expensive.


**Operating System (OS)**

- As always windows operating system are leading in the market
- Apple market share is second. and others maybe linux or Andriod OS.
- Windows OS is holding 96% of Market share.
- Apple Market share is 3%. Maybe of expensive.
- Others OS like linux and andriod maybe more are also holding 3% Market Share.


**Ram Type**

- DDR4 Ram is leading the market.
- DDR5 Ram is second lead just because of recently launch or expensive.
- DDR4 Ram is holding 78% of Market share.
- DDR5 Ram also becoming quiet popular and holding market share is 13%.
- Others Ram like DDR4x, DDR5x and LPDDR5 are also holding 8% of Market share.


**RAM**

- 8 GB Ram are leading the market.
- second most popular is 16 GB Ram.
- 56% of Market share holding by 8 GB Ram just because of base variant of laptop come with 8GB of Ram Size.
- 16 Ram Market share is 37% it also quiet popular.
- Some low cost Laptop comes with 4 GB of Ram and Premium Gaming Laptop comes with 32 GB of Ram Size.


**Processor**

- Intel is still leading the Laptop Market.
- Second lead of the Laptop market is AMD.
- Others comes with Snapdragon processor.
- Intel Processor Market share is 54 %.
- AMD Processor Market share is 43%.


**Graphics Processing Unit (GPU)**

- AMD is leading the GPU market 
- NVIDIA is also lead the GPU market
- Others is Top lead of the GPU Market just because its less costly and its Integrated GPUs.
- AMD GPU Market share is 22% just because its less expensive compare to NVIDIA.
- NVIDIA Market share is 2% because its expensiveness.


**Warranty**

- Majorly Laptop Companies provide only 1 year of Warranty.
- Premium Laptop provide 2 years of Warranty.
- Apple Laptop also comes from 2 years Warranty.
- Also if we pay more Laptop Warranty comes with 2 or more years of Warranty.
- 77% of Laptop Market share hold by 1st year Warranty rest others.


**Screen Size**

- 15.6 Inches of Screen Size of Laptop lead the Market.
- 14.0 Inches of Screen Size of Laptop Second lead the Market.
- 83% of Laptop Market share hold by 15.6 Inches Screen size.
- 17% of Laptop Market share hold by 15.6 Inches Screen size.
- Others Screen size like 13.0, 15.0 and 17.0 and etc also Laptops comes.


**Disk Type**

- SSD Disk Type lead the Laptop Market.
- HDD Disk Type is a second lead of Laptop Market.
- 93% of Laptop Market share is hold by SSD Disk type.
- 7% of Laptop Market share is hold by HDD Disk type.
- HDD Disk type is lossing the market because its low speed.


**Disk Size**

- 512 GB of Disk Size is leading the Laptop Market.
- 1024 GB of Disk Size is second lead of the Laptop Market.
- Others like also 256 GB and 128 GB is holding the Laptop Market share.
- 52% of Laptop Market share is hold by 512 GB.
- 32% of Laptop Market share is hold by 1024 GB.


**Price**

- Mid Range of Laptop is holding More Market share.
- Premium Range of Laptop Market share is Less just because of its expensive.



## Bivariate Analysis

Bivariate analysis is a statistical technique used to analyze the relationship between two variables. It involves examining the distribution of each variable and the relationship between the two variables. Common techniques used in bivariate analysis include scatter plots, correlation analysis, t-tests, and chi-squared tests. Bivariate analysis is useful for exploring relationships among variables and is often the first step in statistical analysis.


**Brand Name VS Price**

<p align="center">
  <img  src="Images\34.PNG">
</p>

- Price depend on Companies premium laptops


**RAM Type VS RAM**

<p align="center">
  <img  src="Images\35.PNG">
</p>


- DDR4 RAM Type Lead the Market


**RAM VS Price**

<p align="center">
  <img  src="Images\36.PNG">
</p>

- RAM Size increase also Price increase.


**Warranty VS Price**

<p align="center">
  <img  src="Images\37.PNG">
</p>

- Warranty depends on Laptop Prices.


**Disk Type VS Price**

<p align="center">
  <img  src="Images\38.PNG">
</p>

- SSD are more expensive than HDD.


**Disk Size VS Price**

<p align="center">
  <img  src="Images\39.PNG">
</p>

- RAM Size depends on Laptops Price.


**Operating System (OS) VS Price**

<p align="center">
  <img  src="Images\40.PNG">
</p>

- MAC OS Laptops are more expensive than others.


**Screen Size VS Price**

<p align="center">
  <img  src="Images\41.PNG">
</p>

- Screen Size depends on Prices.

**Graphics Processing Units (GPU) VS Price**

<p align="center">
  <img  src="Images\42.PNG">
</p>

- NVIDIA GPUs are Expensive than others.


**Processor VS Price**

<p align="center">
  <img  src="Images\43.PNG">
</p>

- Intel Processor are more expensives than others.


## Multivariate Analysis

Multivariate analysis is a statistical technique used to analyze data involving multiple variables, which helps researchers to identify relationships, patterns, and trends among the variables. Some common types of multivariate analyses include multiple regression analysis, principal component analysis, factor analysis, cluster analysis, and discriminant analysis. Multivariate analysis is useful in many fields and can provide insights into complex relationships among variables, improving decision-making and identifying important variables.


**RAM VS Price VS Warranty**

<p align="center">
  <img  src="Images\44.PNG">
</p>

- Warranty and RAM Increase also Laptop Price is Increase.


**Disk Size VS Screen size VS Price**

<p align="center">
  <img  src="Images\45.PNG">
</p>

- Disk Size and Screen size Increase also Laptop Price is Increase.


# Model Building

In summary, the steps involved in building a machine learning model include data preparation, splitting the data into training, validation, and test sets, choosing an appropriate model, training the model on the training data, evaluating the model on the validation and test sets, fine-tuning the model based on the evaluation results, and finally deploying the model for making predictions on new data.



## Train Test Split

Train-test split is a technique used in machine learning to evaluate the performance of a model on new, unseen data. It involves splitting the available data into two subsets - a training set and a test set. The model is trained on the training set and its performance is evaluated on the test set. This allows the model to learn patterns from the training data and be evaluated on how well it can generalize to new data. The goal of train-test split is to ensure that the model is not overfitting the training data and can perform well on new data. Typically, around 70-80% of the data is used for training, and the remaining 20-30% is used for testing. Cross-validation is another technique that can be used in combination with train-test split to further evaluate the model's performance.


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)