# DSI23-Project-4
![GA](http://imgur.com/1ZcRyrc.png)
# General Assembly Data Science Immersive Course Capstone Project

### Finally it is the last project

## Part 1 - Introduction
Weather prediction can be difficult. In the past, humans relied on multiple ideas - shamans, astrology, etc... to tell weather. As technology advanced, barometers and many technological feats were created to study weather patterns to try to know the future. Now, we can combine machine learning techniques and supercomputers to push this even further. However, at the end of the day, time is still not something controlled by humans. A prediction is still a prediction nonetheless. Even a 99.99% accurate prediction has possibility of the opposite happening. How much can humans push the boundaries to foretell or even know for certain what the future weather will be? I do not know.

## Part 2 - Problem Statement
![lol](https://i.pinimg.com/originals/25/24/fd/2524fdcf5a717d7b619fdae3eb314182.gif)
##### Well this project started with an idea. The idea was to bring together a group of remarkable people, see if they could become something more. See if they... Sorry, I am going off topic
It started when a friend approached me on a project he suggested. Use machine learning techniques to reduce carbon footprint of coastal bridges to be built. As I did not have any data at that time, I could only find what I could find online. I considered that there may be a possibility that wind loading could have an impact on building bridges. However, he had help and it was determined that whatever I had done wasn't needed anymore.<br>
So I had to reframe the problem statement from the start again. I had done premature modeling by then. Albeit obtaining horrible results, i did not want the project to go to waste. So as I was working with wind data, I started wondering if Machine Learning models could forecast future wind values accurately?<br>
Could Machine Learning models help create synthetic (future) data for research teams to model weather data into their own research?

## Part 3 - Initial Work
Firstly, the restrictions placed was that the location of the coastal bridges would be built in East Asia (Lower), which puts it around China and Hong Kong. Due to the inconvenience of extracting China meteorology data, I eventually settled for the [Hong Kong Observatory](https://data.gov.hk/en-datasets/category/climate-and-weather?order=name&file-content=no) datasets.<br>
These were the data I pulled
- Max, Min, Mean Temperatures
- UV readings
- Sunshine Readings
- Wind Speed
- Wind Angle

There were data I wished I could pull, like rainfall, precipitation, humidity, etc... Unfortunately, the datasets did not offer them in a daily format for the number of years I needed.
With this, we can proceed to Part 4

## Part 4 - Exploratory Data Analysis Part 1
This isn't exactly Part 1/Part 2 of EDA, but rather, it is one entire modeling which did not yield the results desired, which lead to a whole different modeling later<br>
Due to the multiple number of observatory locations in Hong Kong, I decided to go for 3 locations and take the average amongst them. The criteria for the locations were<br>
a) Offered the most data collected<br>
b) As close to the sea (coastal) areas<br>
c) Looks nice on google maps<br>
<br>
So the 3 places I chose were<br>
a) Cheung Chau<br>
![cch](/img/cch.jpg?raw=true)<br>
b) Hong Kong Airport<br>
![hka](/img/hka.jpg?raw=true)<br>
c) King's Park<br>
![kp](/img/kp.jpg?raw=true)<br>
The feature columns planned out were
- Difference in Temperature between Max and Min temperatures
- Mean Temperature
- UV Readings
- Sunshine Readings
- Wind Speed
- Wind Angle
- Datetime

First, I extracted 2018, 2019, 2020, 2021 data and split them into the individual 12 months (For scaling later, which may be a mistake)<br>
As I was unsure how to use the dates as a feature, and since the data given was in float, I decided to attempt standard scaling and min-max scaling on the dates.<br>
And since each month do not equal number of days, I used the each individual month and did min-max scaling on each day<br>
An example is<br>
______________________________
| 01 | Jan | 2018 -> 0 | 0 | 0 |<br>
______________________________________
| 25 | Dec | 2019 -> 0.8 | 1 | 0.3333 |<br>
_______________________________________

Second, I wanted to check the distribution of my target variable. Initial charting show a nice distribution, but skewed right<br>
![wd](/img/winddist.JPG?raw=true)<br>
So I tried logging the wind values to see if it would give me a better distribution. It did give me a more 'normal' distribution<br>
![wld](/img/windlogdist.JPG?raw=true)<br>

Thirdly, I accounted for seasonality. Using [solar terms](https://www.asia-home.com/china/solterms.php), I used the Chinese solar terms to consider the period of Spring, Summer, Autumn and Winter. I one-hot encoded the values into the respective columns.<br>

Thereafter I concat all the dataframes together to obtain this<br>

![df](/img/df.JPG?raw=true)<br>

A correlation matrix was done thereafter, which yielded... Unpromising characteristics.........
![cm](/img/corrm.JPG?raw=true)<br>

#### Graphing
After including datetime stamps,these are the graphs I got for temperature and windspeed
![temp](/img/temp.JPG?raw=true)<br>
Temperature vs Time<br>
![ws](/img/windspeed.JPG?raw=true)<br>
Wind Speed vs Time<br>
For the windspeed, There are a few outliers which I removed using a condition to drop any value above 40 (3 data points), and given the sparsity of the points, it's quite difficult to define a linear line(?)<br>
Temperature seems to be a cyclic event, so I'm not sure how it will work out at this point<br>

## Part 5 - Basic Modeling
So for the train-test set, it was split into training set is all values before 01/01/2020, and test set consisted of all values after.<br>
Now, as I was trying on scaled data, I had to break it down even further<br>
2 target variables - Wind Speed, Log Wind Speed<br>
3 * [feature columns] - No scaling features, Min-Max features, Standard Scaling features<br>
So a total of 3 * 2 sets were needed that consisted of non-scaled data, min-max scaled data and standard scaled data.<br>

I decided to run a GridSearchCV on the data using these models
- Linear Regression
- Lasso Regression
- Ridge Regression
- GradientBoost
- Adaboost
- Random Forest

I'm not going into further details here, as you can see from the notebook that the results sucked. 

## Part 4 - Exploratory Data Analysis Part 2
So well, it was back to the drawing board again. Somebody once told me the world is gonna roll me... I got lost in my thoughts again. Someone suggested that I could try LSTMs.
And thanks to [Tensorflow](https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_autoregressive_model), I can achieve that.<br>

Step 1: work was done on the wind values. Tensorflow suggested that due to Wind Speed (Scalar propert) and Angles (Degrees), it might be more sensible to work with radian data. So the wind values were converted into a Wx and Wy vectors for manipulation.<br>

Step 2: Also timestamp data was a difficult topic to work with. So a suggestion was to use the data as periodicity. Turn the data into Cosine and Sine Signals.

The following graph shows the 'Year Sine' and 'Year Cosine' Signals. Intepreting days as Sine and Cosine waves.<br>
![sc](/img/sc.JPG?raw=true)<br>

Step 3: Normalizing of values<br>
Using the formula - (x - mean) / std, a simple normalization was done to the values in the dataset. After train-test split of course.<br>
Tensorflow recommendation is 
- train - 70%
- validation - 20%
- test - 10%

The distribution of plots are as indicated below<br>
![wp](/img/violinplot.JPG?raw=true)<br>

## Part 5 - Baseline vs LSTM vs AutoRegressive Models - Modeling Part 2
Well, my entire idea of this project is to model using LSTMs, so although I followed the tutorial and did some work on MLP and CNN, I will not be talking about them or the results in this readme.<br>
Also, I will not be focusing on the specific results of Temperature, Wx or Wy. The model is quite general and I am able to change the target label (variable) as long as it exists in the dataframe column. This will be a generic run-through of how the baseline models compare to LSTMs for single time step, and how baseline compares to AR for multi timestep.

First, to understand and utilize timeseries, we must create a function called WindowGenerator.<br>
![window](/img/window.png?raw=true)<br>
It is best described by the picture above.<br>
##### Just click on the picture link... I don't know why png files turn out like that. I'm not going to take another snip just for you<br>

In this case, input width indicates the 'past' history, and offset indicates the 'future', and label width indicates the prediction. So window generator sets the data to have 24 timesteps of history to learn from to predict a singular value 24 hrs into the 'future'

Baseline Mode<br>
![baseline](/img/baselinemodel.JPG?raw=true)<br>
A graphical representation on how it works. Labels are taken of the 'future' and compared to the predicted values. As this is just a baseline. The predicted values are just shifted right. t=0 -> t=1<br>
![wxbaseline](/img/wxbaseline.JPG?raw=true)<br>
Wx Baseline Model<br>

LSTMs<br>

![rnn](/img/rnn.png?raw=true)<br>
##### Note: Picture not working? Click the picture to view more<br>
The LSTM model shown above takes into consideration the previous timestep (not the output) to model the next predicted value
![wxlstm](/img/wxlstm.JPG?raw=true)<br>
Wx LSTM Model<br>

RMSE results<br>
![rmsess](/img/rmsess.JPG?raw=true)<br>

it seems that the low rmse scores corresponding to the LSTM model show some promise

Let's move to AR models for multi step

Let's first do a baseline<br>
![msbaseline](/img/msbaseline.JPG?raw=true)<br>

Not much can be derived, just a 'copy-paste' to the right

Auto Regressive Model using LSTM<br>
![multistep_autoregressive](/img/multistep_autoregressive.png?raw=true)<br>
##### CLICK THE PICTURE TO SEE THE MODEL<br>
AutoRegressive uses the output of the previous prediction to model the next output. And we will be using this to model the next 30 days<br>

![mmar](/img/mmar.JPG?raw=true)<br>
I'm guess due to the lack of a trend, and many spikes, it doesn't model too well.

Let's do a rmse comparison
![rmsemm](/img/rmsemm.JPG?raw=true)<br>
Needless to say, baseline performed the worst, and the other models do not seem to produce significant results as they can't go below 0.8<br>

## part 6 - Executive Summary
LSTMs work well if predicting single step for weather data, however it may not work as well for multiple steps into the future. More work can be done on this area. Or maybe there was too much varying points to obtain a good prediction.

### To-do List
- EDA
__________
- [ ] Find more data
__________
- Scikit Learn Modeling
__________
- [ ] Drop Datetime as a feature
- [ ] Include more models
__________
- Tensorflow
__________
- [ ] Test different parameters
- [ ] Add more layers
- [ ] Expand on more ideas
__________
- Future plans for this project
__________
- [X] Don't care as I am tired

__________
Sources:
[1]: https://data.gov.hk/en-datasets/category/climate-and-weather?order=name&file-content=no
[2]: https://www.asia-home.com/china/solterms.php
[3]: https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_autoregressive_model
