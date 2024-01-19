# ZILLOW HOUSING DATA TIME SERIES ANALYSIS
# Business Understanding

Real investment firms weigh on many factors when deciding on which investment to make. These firms’ goals are to enhance their decision making processes. With historical real estate market data, the challenge is to leverage data science techniques to identify potential investment opportunities and helping the real estate firms in making informed investment decisions.


## Problem Statement

A Real Estate Investment Firm wants to know the top 5 best zip codes to invest in. As a Data Science consulting group, we have been tasked with finding out.
The task at hand is to create a model that will inform us on the Real Estate investment market trends for the next 10 years.

## Objectives
Main objectives:

* To optimize Investment Performance through Holistic Data-Driven Strategies

Secondary objectives:

* To select the 5 best zipcodes that offer best investment opportunities.
* To predict the value range for the top 5 zipcodes.

## DATA UNDERSTANDING
The dataset contains 14723 rows and 272 columns.    
Each of the row is a unique zipcode.    
The dataset as seen is in a Wide format. Columns 1-7 show the different properties of a house.    
However,  column 8 to column 272 are actual time series values. The columns refer to the median house sales values for their respective month and year.      
This format makes the dataframe intuitive and easy to read. However problems with this dataset may come in when it comes to actually learning from the data. We'll deal with that when we get there.   
The first 7 columns represent:
* RegionID - The Regional ID for the region where a house is located.
* RegionName - The Zipcode. 
* City - The City of a particular house. 
* State - The state in which a home is in. 
* Metro - The metropolitan area where the home is found. 
* CountyName - The county where the home is in. 
* SizeRank - The rank of a property's size relative to other properties in the dataset. 


### Choosing zipcodes
We will use Return On Investment (ROI) and Compound Annual Growth Rate (CAGR) as our metrics  to decide the best zipcodes.          
ROI measures the profitability while CAGR checks the variation.      
We will measure the best zipcodes for the last 10 years.    
We will do this by adding roi column and cagr column to calculate the return on investment and the cagr respectively for every zipcode.  


### EDA AND DATA VISUALIZATION.
We can check the states where these best 5 zipcodes are found

![Alt text](image.png)
Our best zipcodes are found in only three states.           
There are two zipcodes from both CA and NY while PA has one zip code. 

#### Distribution of the Average Home Prices   

![Alt text](image-1.png)

The observation of a bimodal distribution in the data implies the existence of two distinct peaks, indicating that the dataset is not unimodal or normally distributed. In the context of ZipCodes, this suggests the presence of two separate groups or subpopulations within the dataset, each characterized by its own set of home values. These distinct peaks may signify variations in housing characteristics, economic factors, or other influential variables that differentiate the two groups. Analyzing and understanding these subpopulations can be crucial for targeted decision-making, as it allows for a more nuanced exploration of factors influencing home values within different segments of the ZipCode.

#### Time Series  Plots
![Alt text](image-2.png)

HDuring the period from 1996 to 2007, house prices experienced consistent growth. However, the market witnessed a substantial drop from 2007 to 2011, attributed to the profound impact of the housing market crisis in 2008 in the USA. Following the downturn, a subsequent recovery began, marked by a renewed uptrend in house prices, indicating a gradual stabilization and resilience in the real estate market.

#### Testing for Trends

The time series plot above provides a visual indication of a potential linear trend, suggesting a systematic pattern in the data over time. Linear trends can pose challenges for time series models that assume stationarity, as these trends imply a consistent directional movement. To assess the presence of trends more rigorously, a common approach is to employ rolling statistics. These statistical measures, calculated over moving windows, enable the detection of patterns and trends that may not be immediately evident in the overall time series plot. By testing for trends using rolling statistics, we gain a more granular understanding of the data dynamics, helping us make informed decisions about model assumptions and potential adjustments needed for accurate time series forecasting. 


![Alt text](image-3.png)


The observation that the rolling mean is not stationary despite the rolling standard deviation being close to constant suggests that the data may exhibit a non-constant mean over time. Stationarity is a crucial assumption for many time series models, and its violation can impact the model's accuracy. To further assess stationarity, additional tests can be employed.      
By conducting these tests, we aim to gain more insights into the temporal behavior of the data and ascertain whether transformations or adjustments are needed to meet the stationarity requirement for effective time series modeling.

![Alt text](image-4.png)

After differencing the data, observing that it is stationary at a 5% significance level implies that the transformation has effectively removed trends or patterns that rendered the original time series non-stationary. Differencing is a common technique used to stabilize the mean of a time series by subtracting the previous observation from the current one. Achieving stationarity at a 5% significance level means that the transformed data satisfies the criteria for constant mean and variance, enabling the application of various time series models. This result is significant as it validates the efficacy of the differencing process in making the data suitable for modeling, enhancing the accuracy of forecasting and analysis.

### AutoCorrelation Function
The autocorrelation function is a function that represents autocorrelation of a time series as a function of the time lag. 
ACF indicates how similar a value is within a given time series and the previous value. (OR) It measures the degree of the similarity between a given time series and the lagged version of that time series at the various intervals we observed

![Alt text](image-5.png)

### Partial Autocorrelation Function

![Alt text](image-6.png)

The observation that the high correlation ends at lag 2 in both the autocorrelation function (ACF) and partial autocorrelation function (PACF) is indicative of the potential order of an autoregressive model. The gradually decreasing ACF and the sudden drop in the PACF after lag 2 suggest that the time series may be best represented by an Autoregressive Model of order 2, denoted as AR(2). This means that the current value of the time series is linearly dependent on the two most recent observations. The absence of correlation beyond two lags suggests that the impact of earlier observations diminishes significantly, providing valuable information for model selection and forecasting accuracy. Understanding the lag structure is essential for choosing appropriate models and capturing the underlying patterns in the time series data.