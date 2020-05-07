![images/beans.png](images/beans.png)

# COFFEEPEDIA

Ahh... coffee. 

Many of us start each day with a steaming hot cup of coffee, but what do we really know about what makes a coffee good or not?

Coffee quality is evaluated by producers, buyers and licensed quality graders ("Q Graders") by a process known as "cupping", or tasting. Through a process of smelling, brewing and slurping, Q graders rate coffee based on several criteria, including aroma, flavor, aftertaste, acidity, body, balance, uniformity, cleanness, and sweetness. 

Essentially, we want to answer the question...

**What makes a *really* good cup of coffee?**

Americans drink an average of 2 cups of coffee per day. There are more than 15,000 Starbucks in the United States alone. The retail coffee industry in the United States is a booming $5.18 billion business. Coffee future stock price becomes very important in business as it will determine the price of the raw coffee bean. Coffee growers and buyers would have opposite minds to earn profit in their respective businesses-coffee growers would want high prices and coffee buyers would want low prices. 

It boils down to one question...

**What if we can predict the stock price?**

We used several machine learning models from the `Scikit-Learn`, `Tensorflow` and `Keras` libraries to predict coffee quality scores based on several attributes as defined by the Coffee Quality Institute. Coffee is evaluated by licensed quality graders ("Q Graders") based on several criteria, including aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean cup, sweetness, cupper points and defects. We considered these criteria, as well as country of origin, farm region, farm owner, farm name, altitude and processing method to determine which factors most influence the overall quality of a coffee, and to predict coffee quality scores on a set of "testing" data, based on a set of "training" data. We then used historical high, low and opening stock prices to make predictions on closing stock prices using the same methods.

* Coffee Quality Institute: https://www.coffeeinstitute.org/
* Quality Data Source: https://github.com/jldbc/coffee-quality-database
* Price Data Source: https://www.investing.com/commodities/us-coffee-c-historical-data

# Coffee Quality Modeling

Before modeling, we determined which columns will be our X and y values. Because we are predicting quality based on coffee attributes, we decided our y value would be Total Cup Points (basically, the total coffee score) and our X values would be everything else--all grading criteria as well as a few other categorical descriptive attributes. Because some of our X values were categorical in nature (country of origin, farm region, farm owner, farm name, and processing method), we encoded X using Ordinal Encoder before splitting our data into training and testing sets. This method retaineds all values in a single column and assigneds an integer to each unique string, rather than using binary encoding to create two dummy columns for each unique value. We then split the data into training (80%) and testing (20%) sets.

We trained several models on our data--4 linear models (linear regression, lasso, ridge and elasticnet), support vector regression, random forest regression and normal and deep neural networks.

## Linear Models

All the linear models performed quite well, all with over 99% accuracy, and did not require hyperparameter tuning. Predicted scores for each model were plotted against actual scores using Matplotlib subplots to compare them side by side.

## Support Vector Regression

We used `SVR`, a regressor, instead of `SVC`, a classifier, because our y -values were continuous integers, not discrete classes. The initial testing and training scores were quite low, so we used GridSearchCV to retune the model parameters until the scores improved. Hyperparameter tuning was quite successful, as evidenced by the plot comparing predicted/actual scores for the untuned and retuned models.

## Random Forest Regression

Again, we used a regressor (`RandomForestRegressor` instead of `RandomForestClassifier`) because our y -values were continuous integers, not discrete classes. This model did not perform as well as the linear models or the retuned SVR model. Even with hyperparameter tuning, the R2 score decreased. Again, we plotted the predicted/actual scores for the tuned and untuned models.

We did uncover some new information from the random forest regression, however. We were able to answer our initial question of “what makes a really good cup of coffee?” i.e. which features are most important when evaluating coffee quality? We examined determine the most important features used for the model and found that clean cup was the most important feature in predicting impact on coffee quality, followed by flavor and aftertaste.

## Neural Networks

The last models trained on our coffee quality data were two neural networks--deep and normal. We used a linear output activation and `mean_squared_error` and `mean_absolute_error` for our loss functions. The normal neural network performed with a final loss of about 12.26 and a mean absolute error of 2.39, which is an improvement from 3783.52 and 54.11, respectively, on the initial epoch. After adding more hidden layers, the deep neural network performed with an even lower loss of 4.76 and mean absolute error of 1.43.

Again, we plotted the predicted/actual scores for these two models on the same plot for comparison. We also added a diagonal line (coeff=1) to show more clearly where the accurate predictions should lie. We found that while the two models were similar, the deep neural network's outliers were closer to the diagonal line than the normal neural network's outliers, indicating the predictions were more accurate for the deep network.

## Analysis

Finally, we created summary plots for the models. Our first plot shows the actual vs. predicted coffee quality scores for each model, all on the same scale for comparison. It appears that the linear models produced the most accurate predictions. 

When we calculate the percent error for the predictions, we get an even clearer picture of the accuracy—the elasticnet and support vector regression (SVR) models have an error closest to zero. The random forest and neural network models clearly did not perform as well, with much higher (though still barely less than -.15%) percent error. If we remove the worst performing models from the chart, we can see the differences between the other models in slightly greater detail. The elasticnet and SVR models, however, still appear to have performed with similar accuracy.

We can also see that the R2 scores for the linear models are higher than the other models, surprisingly including SVR. 

At face value it appears the models with R2 scores close to 1 are the most accurate at predicting coffee quality scores, but we must also examine the residual plots to be sure. Residuals represent the difference between predicted and actual values. The closer they are to 0, the more accurate the model’s predictions are. Here we confirm that the linear models provided the most accurate predictions for coffee quality. Again, if we remove the neural networks from the comparison and rescale the charts to display the residuals with even more granularity, it is revealed that the elasticnet model had the smallest variance between actual and predicted scores. This combined with the model's high R2 score suggests that the **ElasticNet** model predicted coffee quality scores with the greatest accuracy.

# Coffee Price Modeling

## Coffee Futures Price Trends

Our coffee price data is sourced from Investing.com with a date range from December 27, 1979 to April 24, 2020. It includes closing price (Price), opening price (Open), high price (High), low price (Low), volume, and % change. df.describe() is used to get average, maximum, or minimum of coffee future price. 

The closing, opening, high, and low prices tend to be similar as they show very close average, maximum, and minimum prices. Average stock price over 40 years is ~ $1.24. Maximum stock price is ~$3.00. Minimum stock price is ~$0.415 dollars. There is less than a $2 difference between the prices.

It is fair to say that price may fluctuate over time due to economic reasons, but prices remain relatively stable on any given day. Boxplot, histograms, and price trend graphs are visualized below to assist these findings.

## Linear Models

Before creating any models, we decided which feature to assign to our y values. We decided the most relevant feature to predict would be the closing price, leaving opening, high and low prices to act as our X values. Plotting X and y returns a linear trend, suggesting linear modeling may be a good choice for this data.

Linear regression, Ridge regression, Lasso regression, ElasticNet regression, and Decision tree regression were selected to build models and predict the coffee stock prices. Each of these models returned an accuracy over 0.98, and when actual price and predicted price are plotted against each other it creates a linear graph. The noise in the graphs are so minute, it is negletable.

We trained four linear models on our data: Linear regression, Lasso, Ridge and Elasticnet. All the linear models performed quite well, all with over 99% accuracy, and did not require hyperparameter tuning. Predicted scores for each model were plotted against actual scores using Matplotlib subplots to compare them side by side.

## Analysis

As noted in the coffee data analysis, High price is the most important attribute for predicting future coffee price. Open price is the second attribute predicting future coffee price. The error of each model is lower than 0.0001. R2 of each model is very close to 1. The R2 values of Linear regression and Ridge regression are 0.99935, which are the higher than other models by 0.001. All models would be good to predict the future coffee price, but if almost perfect accuracy is required, then Linear or Ridge regression would be the best choice to use for prediction of coffee future price.