### Assumptions

- When aggregating the data into hourly ranges, should I consider that some quarters might have missing values? Or should I just consider missing values AFTER the aggregation?

    - Nevermind, I will consider missing values BEFORE the aggregation.
---

- Given two gen sources, one with less data recorded than the other, which timestamps should I consider?

    - I will consider the timestamps of the gen source with less data recorded, filter out the rest.

---

- Should I check for outliers?
    
    - IMO, in forecasting outliers are not really relevant. I will not check for outliers until the final dataset is ready.

---

- Feature engineering?

    - Makes no sense since the test data only has green energy and load energy. Not more metadata.

---

- Regression or forecasting approach? Univariate or Multivariate? 

- We will try one regression approach using only the data from the last hour of every region, and another one where we use the forecasting approach where we train a model to generalize on timeseries data.

- On the regression task, I will train the model using only the available window of complete data from the train set, and predict the whole test set as with the forecasting approach.
