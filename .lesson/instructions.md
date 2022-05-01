## Homework: statistical models for time series

### Instructions

* You will provide a main.py file; it is critical that the code runs (i.e. if I press the "Run" button there should be no error message)

* You do not need to write text, just code; please `print` important results (otherwise they are executed but do not get printed)

* You may interact with your friends but DO NOT copy code: if I detect errors that have been copy pasted and that are suspiciously similar across different submission the grade will be penalized

## (1)

Read the file `.lesson/assets/NEW-DATA-1.T15.txt` and put it in a dataframe. I want you to generate a forecast for a few of the last observations of the column "10:Lighting_Comedor_Sensor" and then compare against the actual values. Some tips:

* rename the columns to simpler names
* use `pd.to_datetime` to combine the date and time columns
* plot the actuals
* represent the ACF plot
* split the dataset into train and test, with a split around the last 10% of the observations
* use whichever autoregressive model you want: AR, MA, ARIMA, or autoarima (one is enough but it should make sense)
* use the model to predict the values for the last piece of the observarions (10% or whatever you chose)
* plot both the actuals and the predictions

## (2)
Do the same thing for the file `.lesson/assets/fred_unemployment.csv`.