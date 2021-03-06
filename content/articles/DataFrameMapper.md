Title: DataFrameMapper: A cleaner, more robust way to preprocess data
Date: 2019-04-13 17:00
Tags: python
Slug: dataframemapper
Category: Python

Data must be cleaned and put in a particular shape and form prior to applying machine learning models. This task is referred to as "data preprocessing" and is the first step in any data science and machine learning workflow.

It's no secret that data preprocessing is a dull, mundane activity. Not only does it not require much brain energy (for the most part), but it can also get quite repetitive. To make matters worse, data preprocessing is said to constitute 80% of most data scientists' working time.


<img src="images/DataFrameMapper_1_0.jpeg" alt="dt"/>



One can safely say that not many data scientists enjoy data preprocessing, as demonstrated by the cartoon above.

**Enter DataFrameMapper.** Fortunately, data preprocessing doesn't have to be as tedious as some Kaggle competitors I've seen have made it out to be (!). With DataFrameMapper, code that can clean and transform thousands, millions of rows of data can be written in a very concise, robust and standardized fashion.

Let's dive deeper into what DataFrameMapper is and how it can be used to make data preprocessing a little bit less dreadful.

### What is DataFrameMapper?
----

DataFrameMapper is a module in an experimental scikit-learn library called sklearn-pandas, developed to bridge the gap between actual pandas and scikit-learn, the two most common technologies in a data scientist's toolkit.

The traditional way of doing things in data science is to clean and prepare the data in pandas, and then pass it on to scikit-learn to apply machine learning models. There are modules available in both pandas and scikit-learn to handle common data preprocessing tasks like imputing nulls and turning categorical columns into numeric values. But, without a standardized way to perform these tasks, the procedure of data preprocessing can get quite fragmented and messy, which becomes a problem when new data needs to be fed to a machine learning model.

DataFrameMapper enables all the steps of data preprocessing to be grouped together and stored in a single object, and applied to any dataset with a single operation.

### How does it work?
----

DataFrameMapper maps preprocessing tasks to each column of a given dataset via a list of tuples. Each tuple in the input list refers to a specific column of the dataframe. The first element in the tuple takes the name of the column, and the second element takes the preprocessing task or tasks that want to be applied to that particular column. If there is more than one task, the second element of the tuple needs to be a list, the order of which needs to match the desired order of operations.

Let's see how DataFrameMapper works with an example. First, `pip install sklearn-pandas`, and import it onto your workspace as follows.


<pre class="prettyprint">
#!pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper

# other imports
from IPython.display import HTML
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
</pre>

I'm going to spin up a dataframe of my favorite tv shows, including information on their production cost (made up), number of seasons, mainstream popularity score out of 10 (made up), genre and whether or not they are on netflix.


<pre class="prettyprint">
my_favorite_shows = {
    'name': ['sense_8',
             'handmaidens_tale',
             'the_good_place',
             'big_little_lies',
             'jane_the_virgin',
             'game_of_thrones',
             'mad_men',
             'the_crown',
             'narcos',
             'house_of_cards',
             'girls',
             'breaking_bad',
             'bad_blood',
             'fauda',
             'jessica_jones'],
    'cost': [None,
             140000000,
             80000000,
             170000000,
             205000000,
             600000000,
             300000000,
             None,
             400000000,
             500000000,
             112000000,
             380000000,
             10000000,
             75000000,
             None],
    'seasons': [2, None, 3, 1, 5, 9, 7, 2, 3, 5, 6, 5, 2, None, 2],
    'popularity': [5.8, 6, 5.7, 7.3, 6.5, 9.8, 8.4,
                   7.6, 8, 9.3, 7, 8.9, 2.3, 5.2, 4.7],
    'genre': ['science_fiction',
              'speculative_fiction',
              'comedy',
              'drama',
              'comedy',
              'fantasy',
              'period_drama',
              'period_drama',
              'period_drama',
               None,
              'comedy',
              'crime',
              'crime',
              'crime',
              'science_fiction'],
    'on_netflix': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes',
                   'yes', None, 'yes', 'no', 'yes', None, 'yes', 'yes']
}

my_favorite_shows = pd.DataFrame(my_favorite_shows)
HTML(my_favorite_shows.head(10).to_html(classes="table table-stripped table-hover table-dark"))
</pre>




<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>cost</th>
      <th>seasons</th>
      <th>popularity</th>
      <th>genre</th>
      <th>on_netflix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sense_8</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.8</td>
      <td>science_fiction</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>handmaidens_tale</td>
      <td>140000000.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>speculative_fiction</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>the_good_place</td>
      <td>80000000.0</td>
      <td>3.0</td>
      <td>5.7</td>
      <td>comedy</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>big_little_lies</td>
      <td>170000000.0</td>
      <td>1.0</td>
      <td>7.3</td>
      <td>drama</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>jane_the_virgin</td>
      <td>205000000.0</td>
      <td>5.0</td>
      <td>6.5</td>
      <td>comedy</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>game_of_thrones</td>
      <td>600000000.0</td>
      <td>9.0</td>
      <td>9.8</td>
      <td>fantasy</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mad_men</td>
      <td>300000000.0</td>
      <td>7.0</td>
      <td>8.4</td>
      <td>period_drama</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>the_crown</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>7.6</td>
      <td>period_drama</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>narcos</td>
      <td>400000000.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>period_drama</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>house_of_cards</td>
      <td>500000000.0</td>
      <td>5.0</td>
      <td>9.3</td>
      <td>None</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>



Let's say I want to predict the popularity score of a tv show using its production cost, genre, number of seasons and whether or not it's on netflix. Before I can train a machine learning model with the data I have, I need to get rid of all the NaN values that I intentionally put, and encode all categorical attributes as numbers.

Let's see how preprocessing this dataset would look like *without* using DataFrameMapper.

Splitting our dataset into two - data with which we will train our model and data with which we will test the performance of our model - is the first thing we need to do.


<pre class="prettyprint">
from sklearn.model_selection import train_test_split
X = my_favorite_shows.drop(columns=['name', 'popularity'], axis=1)
y = my_favorite_shows['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
HTML(X_train.head(10).to_html(classes="table table-stripped table-hover table-dark"))
</pre>




<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>seasons</th>
      <th>genre</th>
      <th>on_netflix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>600000000.0</td>
      <td>9.0</td>
      <td>fantasy</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>170000000.0</td>
      <td>1.0</td>
      <td>drama</td>
      <td>no</td>
    </tr>
    <tr>
      <th>9</th>
      <td>500000000.0</td>
      <td>5.0</td>
      <td>None</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>400000000.0</td>
      <td>3.0</td>
      <td>period_drama</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140000000.0</td>
      <td>NaN</td>
      <td>speculative_fiction</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>205000000.0</td>
      <td>5.0</td>
      <td>comedy</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>science_fiction</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>300000000.0</td>
      <td>7.0</td>
      <td>period_drama</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>75000000.0</td>
      <td>NaN</td>
      <td>crime</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>112000000.0</td>
      <td>6.0</td>
      <td>comedy</td>
      <td>no</td>
    </tr>
  </tbody>
</table>



Next, we need to get rid of all the nulls. I'll fill the null values in numerical columns with the median value for the respective column and the nulls in categorical columns with 'unknown'.


<pre class="prettyprint">
median_budget = X_train['cost'].quantile(0.5)
median_season = X_train['seasons'].quantile(0.5)
X_train['cost'] = X_train['cost'].fillna(median_budget)
X_train['genre'] = X_train['genre'].fillna('unknown')
X_train['seasons'] = X_train['seasons'].fillna(median_season)
X_train['on_netflix'] = X_train['on_netflix'].fillna('unknown')
</pre>

I need to transform the genre column to numeric values. A common way to do this is with the `LabelBinarizer` function from sklearn, which creates a column for each unique value in a category, and represents membership with 1s and 0s. (1 for members, 0 for non members)

**IMPORTANT ADVICE**: Do NOT use the `get_dummies()` function from pandas to encode your categorical variables! Things will break apart and your model will not work if the categories in your test data does not match the categories in your training data, which is a **VERY** common occurance!


<pre class="prettyprint">
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
binarized = pd.DataFrame(lb.fit_transform(X_train['genre']), columns=list(lb.classes_))
HTML(binarized.head(5).to_html(classes="table table-stripped table-hover table-dark"))
</pre>




<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>fantasy</th>
      <th>period_drama</th>
      <th>science_fiction</th>
      <th>speculative_fiction</th>
      <th>unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



I now have to add these label binarized columns to my X_train, and remove the original genre column.


<pre class="prettyprint">
X_train.drop(columns=['genre'], axis=1, inplace=True)
Z_train = pd.merge(X_train, binarized, how='left', on = X_train.index)
HTML(Z_train.head(5).to_html(classes="table table-stripped table-hover table-dark"))
</pre>



<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key_0</th>
      <th>cost</th>
      <th>seasons</th>
      <th>on_netflix</th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>fantasy</th>
      <th>period_drama</th>
      <th>science_fiction</th>
      <th>speculative_fiction</th>
      <th>unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>600000000.0</td>
      <td>9.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>170000000.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>500000000.0</td>
      <td>5.0</td>
      <td>yes</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>400000000.0</td>
      <td>3.0</td>
      <td>unknown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>140000000.0</td>
      <td>4.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



Since the index of X_train turned to a column (key_0) during the merge, I need to reset it back to it's original state.


<pre class="prettyprint">
Z_train.set_index('key_0', inplace=True)
HTML(Z_train.head(5).to_html(classes="table table-stripped table-hover table-dark"))
</pre>



<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>seasons</th>
      <th>on_netflix</th>
      <th>comedy</th>
      <th>crime</th>
      <th>drama</th>
      <th>fantasy</th>
      <th>period_drama</th>
      <th>science_fiction</th>
      <th>speculative_fiction</th>
      <th>unknown</th>
    </tr>
    <tr>
      <th>key_0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>600000000.0</td>
      <td>9.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>170000000.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>500000000.0</td>
      <td>5.0</td>
      <td>yes</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>400000000.0</td>
      <td>3.0</td>
      <td>unknown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140000000.0</td>
      <td>4.0</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



I also need to encode the on_netflix column as numbers.


<pre class="prettyprint">
Z_train['on_netflix'] = Z_train['on_netflix'].replace({'no': 0, 'yes': 2, 'unknown': 1})
</pre>

The data is finally ready for modelling. Let's create a simple linear regression model and try to predict popularity scores.


<pre class="prettyprint">
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Z_train, y_train)
train_score = model.score(Z_train, y_train)
print(f'Train score: {train_score}')
</pre>

    Train score: 0.9791971116650907


Apparently, our simple linear regression model is able to predict ~98% of the variability in popularity score. Of course, this score is only based on the train dataset - to evaluate the true performance of our regression model, we need to score it on our test data.

I now have go back and replicate everything I did on the training data on the test data in order to be able to pass it onto my model.


<pre class="prettyprint">
X_test['cost'] = X_test['cost'].fillna(median_budget)
X_test['genre'] = X_test['genre'].fillna('unknown')
X_test['seasons'] = X_test['seasons'].fillna(median_season)
X_test['on_netflix'] = X_test['on_netflix'].fillna('unknown')
binarized = pd.DataFrame(lb.transform(X_test['genre']), columns=list(lb.classes_))
X_test.drop(columns=['genre'], axis=1, inplace=True)
Z_test = pd.merge(X_test, binarized, how='left', on = X_test.index)
Z_test['on_netflix'] = Z_test['on_netflix'].replace({'no': 0, 'yes': 2, 'unknown': 1})
Z_test.set_index('key_0', inplace=True)
</pre>


<pre class="prettyprint">
test_score = model.score(Z_test, y_test)
print(f'Test score: {test_score}')
</pre>

    Test score: 0.43538804012831567



<pre class="prettyprint">
print(f'Predicted scores: {list(model.predict(Z_test))}')
print(f'Actual scores: {list(y_test)}')
</pre>

    Predicted scores: [6.937697253068383, 4.820338983050848, 6.25195791934541]
    Actual scores: [7.6, 2.3, 8.9]


As expected, the model was grossly overfit and the performance of the model on the test dataset is pretty bad. (Not a concern since our data is fake and we are not trying to build an actual model here.)

Let's see how much more easily reproducible data preprocessing would be had we used DataFrameMapper.


<pre class="prettyprint">
from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
mapper = DataFrameMapper([
    (['cost'], SimpleImputer(strategy='median')),
    (['seasons'], SimpleImputer(strategy='median')),
    ('genre', [CategoricalImputer(strategy='constant', fill_value='unknown'),
               LabelBinarizer()]),
    ('on_netflix', [CategoricalImputer(strategy='constant', fill_value='unknown'),
                   LabelEncoder()])
], df_out=True)
Z_train = mapper.fit_transform(X_train)
HTML(Z_train.head(5).to_html(classes="table table-stripped table-hover table-dark"))
</pre>




<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>seasons</th>
      <th>genre_comedy</th>
      <th>genre_crime</th>
      <th>genre_drama</th>
      <th>genre_fantasy</th>
      <th>genre_period_drama</th>
      <th>genre_science_fiction</th>
      <th>genre_speculative_fiction</th>
      <th>genre_unknown</th>
      <th>on_netflix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>600000000.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>170000000.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>500000000.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>400000000.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140000000.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>




<pre class="prettyprint">
Z_test = mapper.transform(X_test)
HTML(Z_test.head(3).to_html(classes="table table-stripped table-hover table-dark"))
</pre>




<table border="1" class="dataframe table table-stripped table-hover table-dark">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
      <th>seasons</th>
      <th>genre_comedy</th>
      <th>genre_crime</th>
      <th>genre_drama</th>
      <th>genre_fantasy</th>
      <th>genre_period_drama</th>
      <th>genre_science_fiction</th>
      <th>genre_speculative_fiction</th>
      <th>genre_unknown</th>
      <th>on_netflix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>187500000.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10000000.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>380000000.0</td>
      <td>5.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>



**That's it, friends!** We were able to do everything that was done previously with just 5-6 lines of code. The best part is that now, I can transform any new data to a model-ready state with a single line of code.

Let's pretend like we have to predict the popularity score of a new show, not included in our original dataset. Without DataFrameMapper, we would have to repeat the previous preprocessing steps for a third time. With DataFrameMapper, we can just pass the new data onto mapper.transform, and immediately get a prediction.


<pre class="prettyprint">
new_data = {'name': ['the_protector'],
            'cost': [5_000_000],
            'seasons': [1],
            'genre': ['science_fiction'],
            'on_netflix': ['yes']}
new_data = pd.DataFrame(new_data)
new_Z = mapper.transform(new_data)
print(f'Predicted popularity score: {round(float(model.predict(new_Z)), 3)}')
</pre>

    Predicted popularity score: 9.885


----
#### Notice that:

* We fit transform the training data, but only transform the test data and the data for which we want to get a prediction.

* The `df_out=True` argument enables us to get a dataframe output from the transform function. By default, `df_out` is set to `False`, so if we don't include it in the mapper we would get a numpy array as the output. Either is fine as far modelling goes, it's more a matter of convenience. I personally prefer seeing pandas dataframes over numpy arrays as I find them easier to read.

* The `LabelEncoder` transformer replaces categorical variables with numerical labels, like the `pd.replace` function used previously.

* When using `SimpleImputer` - which is a sklearn imputation transformer for numeric values - we have to wrap the first element of the tuple, i.e., the column name, within brackets. Otherwise, the mapper will throw an error. This has to do with the fact that `SimpleImputer` needs to take lists as input.

* I passed two transformers to the genre and on_netflix columns - the `CategoricalImputer` first, followed by the `LabelBinarizer` in the case of genre and `LabelEncoder` in the case of on_netflix. If I had done the reverse, the mapper would throw an error because null values can't be label binarized. **As a rule of thumb and general good practice, imputers need to be the first transformation in a mapper and they need to be applied to *all* columns.**

* I imported `CategoricalImputer` from `pandas` whereas I imported `SimpleImputer` from `sklearn.impute`. This is another great thing about `sklearn-pandas`: they provide functionality for the imputation of categorical values, which traditionally did not exist in sklearn.

* The strategy argument in the imputation transformers lets us decide *what* we want to replace the null values with. Available strategies in `SimpleImputer` are median, mean, mode or any constant value of choice. Available strategies in `CategoricalImputer` are the most frequent value or a constant of choice. If strategy is set to constant, the `fill_value` argument also needs to be defined, as I have done above.

### Happy mappin'! :)
