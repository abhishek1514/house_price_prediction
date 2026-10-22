California Housing Price Prediction
Project Overview
This project explores the California Housing dataset and builds
regression models to estimate median house values from census-block
housing and demographic features. The notebook performs exploratory data
analysis (EDA), visualizes feature relationships, prepares training and
test sets, scales features, and evaluates regression models.
Objectives
- Load the California Housing dataset from scikit-learn.
- Explore dataset structure, data types, unique values, duplicates,
  and skewness.
- Visualize feature distributions and the target variable.
- Study correlations between features and median house value.
- Train regression models and evaluate their performance on a held-out
  test set.
Dataset
The notebook uses California Housing, loaded directly from
sklearn.datasets.fetch_california_housing().

The dataset is fetched by scikit-learn, so a separate CSV file is not
required. The target column created in the notebook is MedHouseVal,
the dataset's median house value target (in units of $100,000).

The features are: - MedInc --- median income - HouseAge --- median
house age - AveRooms --- average rooms per household - AveBedrms ---
average bedrooms per household - Population --- block-group
population - AveOccup --- average household occupancy - Latitude and
Longitude --- geographic coordinates
Models Used
- RandomForestRegressor with 300 trees (n_estimators=300).
- HistGradientBoostingRegressor with 300 boosting iterations,
  learning rate 0.05, and maximum depth 6.
The notebook also imports mean_squared_error, but currently reports
the Random Forest R² score and the HistGradientBoosting model's test-set
score (R²). It does not currently print RMSE or MAE.
Exploratory Data Analysis
The notebook includes: - Dataset preview, shape, column names, data
types, and descriptive information. - Unique-value counts and
duplicate-row check. - Feature skewness. - Histograms of all features. -
Target distribution and boxplot. - Correlation heatmap and
feature-to-target correlations. - Scatter plots of each feature against
MedHouseVal.
Workflow
1. Load libraries and fetch the California Housing dataset.
2. Create a Pandas DataFrame and add the MedHouseVal target.
3. Inspect and visualize the data.
4. Separate input features (X) and target (y).
5. Split the data into training and test sets using an 80/20 split.
6. Fit StandardScaler on the training features and transform both
   sets.
7. Train a Random Forest regression model and calculate its R² score.
8. Train a HistGradientBoosting regression model and evaluate its score
   on the test set.
Technologies Used
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
Installation
Install the required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
How to Run
1. Clone or download this repository.
2. Install the dependencies above.
3. Start Jupyter Notebook:
   jupyter notebook
4. Open house_prediction.ipynb.
5. Run the notebook cells in order. The first run may need an internet
   connection to fetch the dataset through scikit-learn.
Evaluation
The notebook reports the R² score on the held-out test data. R²
measures how much of the variation in the target is explained by the
model; closer to 1 is generally better. The actual score depends on the
environment and library versions, so no fixed score is claimed here.

For a more complete evaluation, consider reporting MAE and RMSE
alongside R². Also note that the notebook scales features before
training; tree-based models such as Random Forest and
HistGradientBoosting generally do not require feature scaling, although
using a scaler here is valid.
Limitations and Improvements
- Add MAE and RMSE so prediction errors are easier to interpret.
- Compare models using the same evaluation metrics and test split.
- Add a feature-importance analysis for the Random Forest model.
- Save the trained model and preprocessing pipeline if predictions
  will be used outside the notebook.
- Keep the notebook's final evaluation cells clearly separated for
  each model.
Author
Abhishek babu
