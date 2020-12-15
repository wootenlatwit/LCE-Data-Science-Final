# LCE-Data-Science-Final

# Introduction
The objective of this project is to predict stock data, thus reducing the amount of guesswork needed to turn a profit on stocks. Having stocks in certain companies already, we would love to get a working model so we know when to buy and sell stock. Hopefully if we get our model to predict known data fairly well, we'll be able to have it predict future stocks just as well.
# Selection of Data
The data we selected was from [this site](https://polygon.io/). We took that data that was given to us, which was weekly, and decided to fill it so we could get data that was by the minute. This led to our data set being over 820k data points. Out of those 820k points, we trained 75% of it. With this large data set, we were able to produce a line graph that represented the DOW stock market failry well.
# Methods
Tools:
* Numpy, Scikit-learn, and Pandas for analysis
* Github for communication and sharing
* VS Code as IDE along with Jupyter

Methods from Scikit:
* Linear regression model (SGD Regressor)
* Pipeline
* Standard Scalar

# Results
Looking at the history of a stock's value is just as good as regression model because you can see a trend within the graph. This means it we could identify a stocks value without a regression line in the graph. To improve upon our model, we would like to take regression models for each day and fit that to our model. This could lead to analysis we weren't inherently aware of. 
# Discussion
After working with many regression models, we found that SGD did the best at predicting something that wasn't just a linear model. Our model, for the data it does predict for, actually jumps a little at the beginning before going into a fairly shallow exponential growth model. Although this model doesn't predict the values of stocks very well, it can identify trends in the data. Looking online at other's work on the same issue, we found that they used time series analysis along with tensor flow to implement neural networks for their models. Their accuracy levels were a lot higher than ours, so maybe implementing a neural netwrok in the future would help out our model a lot. 
# Summary
The program created uses SGD regression to predict stock data. The trends in the actual data are easy to identify. However, it is hard to predict exact values of stocks becuase of their unpredictable nature. Our model can only predict a trend within a small portion of the entire stock data.  
