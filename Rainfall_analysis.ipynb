{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1>Rain in Australia</h1>\n",
    "    \n",
    "<h2>Description</h2>\n",
    "\n",
    "\n",
    "This dataset contains daily weather observations from numerous Australian weather stations.    \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "<h2>Features description</h2>\n",
    "\n",
    "    Date---The date of observation\n",
    "    \n",
    "    Location---The common name of the location of the weather station\n",
    "    \n",
    "    MinTemp---The minimum temperature in degrees celsius\n",
    "    \n",
    "    MaxTemp---The maximum temperature in degrees celsius\n",
    "    \n",
    "    Rainfall---The amount of rainfall recorded for the day in mm\n",
    "    \n",
    "    Evaporation---The so-called Class A pan evaporation (mm) in the 24 hours to 9am\n",
    "    \n",
    "    Sunshine---The number of hours of bright sunshine in the day.\n",
    "    \n",
    "    WindGustDir---The direction of the strongest wind gust in the 24 hours to midnight\n",
    "    \n",
    "    WindGustSpeed---The speed (km/h) of the strongest wind gust in the 24 hours to midnight\n",
    "    \n",
    "    WindDir9am---Direction of the wind at 9am\n",
    "    \n",
    "    WindDir3pm---Direction of the wind at 3pm\n",
    "    \n",
    "    WindSpeed9am---Wind speed (km/hr) averaged over 10 minutes prior to 9am\n",
    "    \n",
    "    WindSpeed3pm---Wind speed (km/hr) averaged over 10 minutes prior to 3pm\n",
    "    \n",
    "    Humidity9am---Humidity (percent) at 9am\n",
    "    \n",
    "    Humidity3pm---Humidity (percent) at 3pm\n",
    "    \n",
    "    Pressure9am---Atmospheric pressure (hpa) reduced to mean sea level at 9am\n",
    "    \n",
    "    Pressure3pm---Atmospheric pressure (hpa) reduced to mean sea level at 3pm\n",
    "    \n",
    "    Cloud9am---Fraction of sky obscured by cloud at 9am. This is measured in \"oktas\", which are a unit of                    eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates                    completely clear sky whilst an 8 indicates that it is completely overcast.\n",
    "    \n",
    "    Cloud3pm---Fraction of sky obscured by cloud (in \"oktas\": eighths) at 3pm. See Cload9am for a description                of the values\n",
    "    \n",
    "    Temp9am---Temperature (degrees C) at 9am\n",
    "    \n",
    "    \n",
    "    Temp3pm---Temperature (degrees C) at 3pm\n",
    "\n",
    "    RainToday---Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0\n",
    "\n",
    "    RISK_MM---The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of  measure of the \"risk\".\n",
    "    \n",
    "    \n",
    "    RainTomorrow---The target variable. Did it rain tomorrow?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>References</h2>\n",
    "This total dataset is taken from following reference. \n",
    "\n",
    "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package\n",
    "\n",
    "<h2>Source & Acknowledgements</h2>\n",
    "\n",
    "Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data. Copyright Commonwealth of Australia 2010, Bureau of Meteorology.\n",
    "\n",
    "Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml\n",
    "\n",
    "This dataset is also available via the R package rattle.data and at https://rattle.togaware.com/weatherAUS.csv. Package home page: http://rattle.togaware.com. Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.\n",
    "\n",
    "And to see some great examples of how to use this data: https://togaware.com/onepager/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>Objective</h2>\n",
    "\n",
    "Predict whether or not it will rain tomorrow by training a binary classification model on target RainTomorrow.\n",
    "\n",
    "The target variable RainTomorrow means: Did it rain the next day? Yes or No."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#For Data loading and preprocessing\n",
    "import pandas as pd\n",
    "\n",
    "#For matrix operations\n",
    "import numpy as np\n",
    "\n",
    "#For plotting\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "\n",
    "#For splitting the data\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "#For data preprocessing\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "#For hyperparameter tuning\n",
    "from sklearn.model_selection import RandomizedSearchCV,GridSearchCV\n",
    "#For appling LogisticRegression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "#For model/vatiable persistence \n",
    "from sklearn.externals import joblib\n",
    "\n",
    "#For math operations\n",
    "import math \n",
    "\n",
    "#To see the progress of the iterations\n",
    "#from tqdm import tqdm\n",
    "\n",
    "#Performance metrices\n",
    "from sklearn.metrics import roc_auc_score,roc_curve,auc,log_loss,confusion_matrix\n",
    "\n",
    "#For encoding the features\n",
    "from sklearn.preprocessing import LabelEncoder,LabelBinarizer\n",
    "\n",
    "#For ignoring warnings\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total no.of points = 142193\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Location</th>\n",
       "      <th>MinTemp</th>\n",
       "      <th>MaxTemp</th>\n",
       "      <th>Rainfall</th>\n",
       "      <th>Evaporation</th>\n",
       "      <th>Sunshine</th>\n",
       "      <th>WindGustDir</th>\n",
       "      <th>WindGustSpeed</th>\n",
       "      <th>WindDir9am</th>\n",
       "      <th>...</th>\n",
       "      <th>Humidity3pm</th>\n",
       "      <th>Pressure9am</th>\n",
       "      <th>Pressure3pm</th>\n",
       "      <th>Cloud9am</th>\n",
       "      <th>Cloud3pm</th>\n",
       "      <th>Temp9am</th>\n",
       "      <th>Temp3pm</th>\n",
       "      <th>RainToday</th>\n",
       "      <th>RISK_MM</th>\n",
       "      <th>RainTomorrow</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2008-12-01</td>\n",
       "      <td>Albury</td>\n",
       "      <td>13.4</td>\n",
       "      <td>22.9</td>\n",
       "      <td>0.6</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>W</td>\n",
       "      <td>44.0</td>\n",
       "      <td>W</td>\n",
       "      <td>...</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1007.7</td>\n",
       "      <td>1007.1</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>16.9</td>\n",
       "      <td>21.8</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2008-12-02</td>\n",
       "      <td>Albury</td>\n",
       "      <td>7.4</td>\n",
       "      <td>25.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>WNW</td>\n",
       "      <td>44.0</td>\n",
       "      <td>NNW</td>\n",
       "      <td>...</td>\n",
       "      <td>25.0</td>\n",
       "      <td>1010.6</td>\n",
       "      <td>1007.8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>17.2</td>\n",
       "      <td>24.3</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2008-12-03</td>\n",
       "      <td>Albury</td>\n",
       "      <td>12.9</td>\n",
       "      <td>25.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>WSW</td>\n",
       "      <td>46.0</td>\n",
       "      <td>W</td>\n",
       "      <td>...</td>\n",
       "      <td>30.0</td>\n",
       "      <td>1007.6</td>\n",
       "      <td>1008.7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>23.2</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2008-12-04</td>\n",
       "      <td>Albury</td>\n",
       "      <td>9.2</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NE</td>\n",
       "      <td>24.0</td>\n",
       "      <td>SE</td>\n",
       "      <td>...</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1017.6</td>\n",
       "      <td>1012.8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>18.1</td>\n",
       "      <td>26.5</td>\n",
       "      <td>No</td>\n",
       "      <td>1.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2008-12-05</td>\n",
       "      <td>Albury</td>\n",
       "      <td>17.5</td>\n",
       "      <td>32.3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>W</td>\n",
       "      <td>41.0</td>\n",
       "      <td>ENE</td>\n",
       "      <td>...</td>\n",
       "      <td>33.0</td>\n",
       "      <td>1010.8</td>\n",
       "      <td>1006.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>29.7</td>\n",
       "      <td>No</td>\n",
       "      <td>0.2</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \\\n",
       "0  2008-12-01   Albury     13.4     22.9       0.6          NaN       NaN   \n",
       "1  2008-12-02   Albury      7.4     25.1       0.0          NaN       NaN   \n",
       "2  2008-12-03   Albury     12.9     25.7       0.0          NaN       NaN   \n",
       "3  2008-12-04   Albury      9.2     28.0       0.0          NaN       NaN   \n",
       "4  2008-12-05   Albury     17.5     32.3       1.0          NaN       NaN   \n",
       "\n",
       "  WindGustDir  WindGustSpeed WindDir9am  ... Humidity3pm  Pressure9am  \\\n",
       "0           W           44.0          W  ...        22.0       1007.7   \n",
       "1         WNW           44.0        NNW  ...        25.0       1010.6   \n",
       "2         WSW           46.0          W  ...        30.0       1007.6   \n",
       "3          NE           24.0         SE  ...        16.0       1017.6   \n",
       "4           W           41.0        ENE  ...        33.0       1010.8   \n",
       "\n",
       "   Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  RISK_MM  \\\n",
       "0       1007.1       8.0       NaN     16.9     21.8         No      0.0   \n",
       "1       1007.8       NaN       NaN     17.2     24.3         No      0.0   \n",
       "2       1008.7       NaN       2.0     21.0     23.2         No      0.0   \n",
       "3       1012.8       NaN       NaN     18.1     26.5         No      1.0   \n",
       "4       1006.0       7.0       8.0     17.8     29.7         No      0.2   \n",
       "\n",
       "   RainTomorrow  \n",
       "0            No  \n",
       "1            No  \n",
       "2            No  \n",
       "3            No  \n",
       "4            No  \n",
       "\n",
       "[5 rows x 24 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('weatherAUS.csv')\n",
    "\n",
    "print(\"Total no.of points = {}\".format(data.shape[0]))\n",
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>1. Exploratory Data Analysis</h2> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop_duplicates(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date             False\n",
       "Location         False\n",
       "MinTemp           True\n",
       "MaxTemp           True\n",
       "Rainfall          True\n",
       "Evaporation       True\n",
       "Sunshine          True\n",
       "WindGustDir       True\n",
       "WindGustSpeed     True\n",
       "WindDir9am        True\n",
       "WindDir3pm        True\n",
       "WindSpeed9am      True\n",
       "WindSpeed3pm      True\n",
       "Humidity9am       True\n",
       "Humidity3pm       True\n",
       "Pressure9am       True\n",
       "Pressure3pm       True\n",
       "Cloud9am          True\n",
       "Cloud3pm          True\n",
       "Temp9am           True\n",
       "Temp3pm           True\n",
       "RainToday         True\n",
       "RISK_MM          False\n",
       "RainTomorrow     False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().any()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### We can see there are many Null values in the data , lets try to fill  with proper values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "sns.countplot(data.RainTomorrow)\n",
    "plt.title(\"Target labels\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### From above plot it is clear that data set is imbalanced "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Separating the data based on its class label.\n",
    "data_yes = data[data['RainTomorrow']=='Yes']\n",
    "data_no = data[data['RainTomorrow']=='No']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Now lets observe the most occuring values in every  column for both the cases "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Location</th>\n",
       "      <th>MinTemp</th>\n",
       "      <th>MaxTemp</th>\n",
       "      <th>Rainfall</th>\n",
       "      <th>Evaporation</th>\n",
       "      <th>Sunshine</th>\n",
       "      <th>WindGustDir</th>\n",
       "      <th>WindGustSpeed</th>\n",
       "      <th>WindDir9am</th>\n",
       "      <th>...</th>\n",
       "      <th>Humidity3pm</th>\n",
       "      <th>Pressure9am</th>\n",
       "      <th>Pressure3pm</th>\n",
       "      <th>Cloud9am</th>\n",
       "      <th>Cloud3pm</th>\n",
       "      <th>Temp9am</th>\n",
       "      <th>Temp3pm</th>\n",
       "      <th>RainToday</th>\n",
       "      <th>RISK_MM</th>\n",
       "      <th>RainTomorrow</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2010-09-03</td>\n",
       "      <td>Portland</td>\n",
       "      <td>9.6</td>\n",
       "      <td>17.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>W</td>\n",
       "      <td>39.0</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>69.0</td>\n",
       "      <td>1014.0</td>\n",
       "      <td>1010.4</td>\n",
       "      <td>8.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>13.8</td>\n",
       "      <td>16.0</td>\n",
       "      <td>No</td>\n",
       "      <td>1.2</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date  Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \\\n",
       "0  2010-09-03  Portland      9.6     17.2       0.0          4.0       0.0   \n",
       "\n",
       "  WindGustDir  WindGustSpeed WindDir9am  ... Humidity3pm  Pressure9am  \\\n",
       "0           W           39.0          N  ...        69.0       1014.0   \n",
       "\n",
       "   Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  RISK_MM  \\\n",
       "0       1010.4       8.0       8.0     13.8     16.0         No      1.2   \n",
       "\n",
       "   RainTomorrow  \n",
       "0           Yes  \n",
       "\n",
       "[1 rows x 24 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Observing the mode for all columns when RainTomorrow = Yes  \n",
    "mode_values_for_yes = data_yes.mode()\n",
    "mode_values_for_yes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Location</th>\n",
       "      <th>MinTemp</th>\n",
       "      <th>MaxTemp</th>\n",
       "      <th>Rainfall</th>\n",
       "      <th>Evaporation</th>\n",
       "      <th>Sunshine</th>\n",
       "      <th>WindGustDir</th>\n",
       "      <th>WindGustSpeed</th>\n",
       "      <th>WindDir9am</th>\n",
       "      <th>...</th>\n",
       "      <th>Humidity3pm</th>\n",
       "      <th>Pressure9am</th>\n",
       "      <th>Pressure3pm</th>\n",
       "      <th>Cloud9am</th>\n",
       "      <th>Cloud3pm</th>\n",
       "      <th>Temp9am</th>\n",
       "      <th>Temp3pm</th>\n",
       "      <th>RainToday</th>\n",
       "      <th>RISK_MM</th>\n",
       "      <th>RainTomorrow</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2014-03-18</td>\n",
       "      <td>Canberra</td>\n",
       "      <td>11.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>E</td>\n",
       "      <td>35.0</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1017.9</td>\n",
       "      <td>1015.5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>20.0</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date  Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \\\n",
       "0  2014-03-18  Canberra     11.0     20.0       0.0          4.0      11.0   \n",
       "\n",
       "  WindGustDir  WindGustSpeed WindDir9am  ... Humidity3pm  Pressure9am  \\\n",
       "0           E           35.0          N  ...        52.0       1017.9   \n",
       "\n",
       "   Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  RISK_MM  \\\n",
       "0       1015.5       1.0       1.0     16.0     20.0         No      0.0   \n",
       "\n",
       "   RainTomorrow  \n",
       "0            No  \n",
       "\n",
       "[1 rows x 24 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Observing the mode for all columns when RainTomorrow = No  \n",
    "mode_values_for_no = data_no.mode()\n",
    "mode_values_for_no"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "23.3"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_no['MaxTemp'].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#For Temparatures we cannot replace NaN values with 0, hence replacing NaN with its respective mode value\n",
    "data_yes['MinTemp'].fillna(value=data_yes['MinTemp'].mode()[0],inplace=True )\n",
    "data_no['MinTemp'].fillna(value=data_no['MinTemp'].mode()[0],inplace=True )\n",
    "\n",
    "data_yes['MaxTemp'].fillna(value=data_yes['MaxTemp'].mode()[0],inplace=True )\n",
    "data_no['MaxTemp'].fillna(value=data_no['MaxTemp'].mode()[0],inplace=True )\n",
    "\n",
    "\n",
    "data_yes['Temp9am'].fillna(value=data_yes['Temp9am'].mode()[0],inplace=True )\n",
    "data_no['Temp9am'].fillna(value=data_no['Temp9am'].mode()[0],inplace=True )\n",
    "\n",
    "data_yes['Temp3pm'].fillna(value=data_yes['Temp3pm'].mode()[0],inplace=True )\n",
    "data_no['Temp3pm'].fillna(value=data_no['Temp3pm'].mode()[0],inplace=True )\n",
    "\n",
    "\n",
    "# For humidity also \n",
    "data_yes['Humidity9am'].fillna(value=data_yes['Humidity9am'].mode()[0],inplace=True )\n",
    "data_no['Humidity9am'].fillna(value=data_no['Humidity9am'].mode()[0],inplace=True )\n",
    "\n",
    "\n",
    "\n",
    "data_yes['Humidity3pm'].fillna(value=data_yes['Humidity3pm'].mode()[0],inplace=True )\n",
    "data_no['Humidity3pm'].fillna(value=data_no['Humidity3pm'].mode()[0],inplace=True )\n",
    "\n",
    "# For the rain fall feature we can replace NaN with 0.0 which says there is no rain fall\n",
    "data_yes['Rainfall'].fillna(value=0.0,inplace=True)\n",
    "data_no['Rainfall'].fillna(value=0.0,inplace=True)\n",
    "\n",
    "\n",
    "data_yes['Pressure9am'].fillna(value=data_yes['Pressure9am'].median(),inplace=True )\n",
    "data_no['Pressure9am'].fillna(value=data_no['Pressure9am'].median(),inplace=True )\n",
    "\n",
    "data_yes['Pressure3pm'].fillna(value=data_yes['Pressure3pm'].median(),inplace=True )\n",
    "data_no['Pressure3pm'].fillna(value=data_no['Pressure3pm'].median(),inplace=True )\n",
    "\n",
    "\n",
    "data_yes['WindSpeed9am'].fillna(value=data_yes['WindSpeed9am'].median(),inplace=True )\n",
    "data_no['WindSpeed9am'].fillna(value=data_no['WindSpeed9am'].median(),inplace=True )\n",
    "\n",
    "data_yes['WindSpeed3pm'].fillna(value=data_yes['WindSpeed3pm'].median(),inplace=True )\n",
    "data_no['WindSpeed3pm'].fillna(value=data_no['WindSpeed3pm'].median(),inplace=True )\n",
    "\n",
    "#WindGustSpeed -- replacing with median\n",
    "data_yes['WindGustSpeed'].fillna(value=data_yes['WindGustSpeed'].median(),inplace=True)\n",
    "data_no['WindGustSpeed'].fillna(value=data_no['WindGustSpeed'].median(),inplace=True)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For RainToday feature we cannot fill any value, so better to remove the NaN values \n",
    "data_yes.dropna(inplace=True)\n",
    "data_no.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_filled= data_yes.append(data_no, ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Date             False\n",
       "Location         False\n",
       "MinTemp          False\n",
       "MaxTemp          False\n",
       "Rainfall         False\n",
       "Evaporation      False\n",
       "Sunshine         False\n",
       "WindGustDir      False\n",
       "WindGustSpeed    False\n",
       "WindDir9am       False\n",
       "WindDir3pm       False\n",
       "WindSpeed9am     False\n",
       "WindSpeed3pm     False\n",
       "Humidity9am      False\n",
       "Humidity3pm      False\n",
       "Pressure9am      False\n",
       "Pressure3pm      False\n",
       "Cloud9am         False\n",
       "Cloud3pm         False\n",
       "Temp9am          False\n",
       "Temp3pm          False\n",
       "RainToday        False\n",
       "RISK_MM          False\n",
       "RainTomorrow     False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_filled.isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Percentage of removed points= 60.14642070988023%\n"
     ]
    }
   ],
   "source": [
    "print(\"Percentage of removed points= {}%\".format(100.00-(len(data_filled)*100/len(data))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sorting the data based on data (Time based splitting)\n",
    "data_filled=data_filled.sort_values(by='Date')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Removing unwanted features, RISK_MM is same as target label hence removing with data and loaction  \n",
    "data_final = data_filled.drop(['Date', 'Location','RISK_MM'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(56669, 21)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_final.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Now lets check for any outliers "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025bdd64a8>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Outliers we are checking only for numerical features\n",
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['MinTemp','MaxTemp','Temp9am','Temp3pm']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### From the above box plot we can see that all temparature values are meaning full. no outliers found here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2026580f898>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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XXcpMzNTp0+fbtXxHnroIZWXl0uS/vOf/+iBBx7QhAkTdPfdd2vKlCn6/PPPz7vG7du369NPP5Uk7dy5U+np6br33ns1efJkrV279ryP1xpLlizRxo0b/XJstD1/zvPVq1frnnvuUXp6ujIyMrRr1y7f/wCSkpOTJX37e/Xb3/5WqampevTRR1VXV+eX86H9CaZ5fvz4cU2ePFmpqal68MEHmec+QKDysxEjRugf//iHpG/33BoxYoTee+89SVJDQ4MOHz6s5cuXKyIi4ryOW1dXp6lTp+ree+/Vyy+/rBdeeEG///3vNXv27POu8bXXXtOxY8ckSbNnz9bjjz+udevW6bnnntObb76pf/3rX+d9TIQWf83zPXv2aMuWLVq3bp3Wrl2rRx99VDNnzvR1+c0sXrxYkyZNUnFxsZKSkrRu3Tq/ng/tRzDN89WrV+u2225TcXGx+vbtqw0bNvj1fKHA8o09g93w4cO1YsUKpaen6/3339f48eO1ZMkSVVdXq7KyUomJibrxxhv1l7/8Rbm5uYqIiNDBgwd17NgxLVy4UD/+8Y+1fv16vfLKK+revbtOnDghSXr33Xc1dOhQXX/99Z5zXXPNNXrhhRckSTNmzNAvf/lLpaSkqLS0VG+99ZYWLlyoGTNmaP/+/WpoaFBGRoZ69+6tDz74QJWVlerbt6969uyp9evX6/bbb1f//v314osvKiIiQhs3btTmzZvlcDh06tQpZWZmatSoUdq2bZueeuophYeHq1evXp5Al5ubq3379snlcikrK0tJSUn661//qpUrVyouLk5Op1Px8fFt/x8Cv/DXPI+Li9OhQ4f06quvKiUlRf3799err74qSUpLS1OfPn305Zdfyu1266mnnlL37t315JNPavv27XK73brnnns0ZswYffbZZ5o7d64kqUuXLpo/f76ioqKUnZ2tPXv2qFevXp6uwp49ezRnzhxJ0uDBgzV//nxJ0pNPPqldu3appqZGV155pRYsWKBly5Zp3759OnXqlL7++mulpqbqb3/7m7788kstWrRI1113XVv/V8CPgmmez5w5U263Wy6XS4cPH9aPfvQjSdLPf/5zXXvttdq/f7/69eunefPmafny5czzVqBD5WcDBgzQF198Ibfbre3btysxMVHDhg3T1q1btW3bNo0cObLZ9/fs2VMFBQVKS0vThg0bVF1drRdeeEEvv/yyVqxYIafTKUk6cOCAevfu7fl3U6dOVVpamkaPHq0jR460WIvD4VB5ebmeeeYZrVmzRk1NTRo4cKBGjhypxx57TD179tT8+fPVrVs35eXlafjw4Vq0aJHnF7C2ttbzCmrhwoVyOp3Kzs7WM888o6KiIvXo0UOvv/66XnnlFXXt2lXr16/XihUrPCFr8eLFWrdunQoKCtSxY0d/PNywiL/meVxcnFauXKkdO3Zo4sSJGj16tN59913PcQYPHqzCwkKNGTNGzz77rN5//30dOHBAL730kl544QWtWrVK33zzjbKzs5Wbm6vCwkKlpKToueeeU2lpqRoaGvTyyy/rkUce8Vzy6N+/v7Zs2SJJ2rx5s+rq6uRwOBQbG6t169bppZde0scff6yjR49Kkjp27KiCggLdfPPNev/997Vq1SpNmTJFb775Zls89GhDwTTPbTabmpqa9Otf/1rl5eUaPHiwJOno0aN68MEH9eqrr6q2tlbvvPOOJOZ5a9Ch8rOwsDBdffXVKi0tVffu3RUREaGUlBS99957+vTTT3X33Xc3+/7+/ftLki655BLt2LFDX3zxhfr27etpIV9zzTWer599jX3lypWSpAkTJqixsbHZMc9shh8TE6Ps7GxlZ2fL4XDolltuafZ9DQ0NqqysVGZmpjIzM3Xq1CnNnDlTGzZsUHR0tG644QaFhYXpoosuUmxsrI4dO6Zjx44pKytLklRfX6/k5GR99dVXstvtnrUGjY2NqqqqUkxMjLp27SpJzTpraP/8Nc/37dunmJgYLViwQJJUUVGhKVOmKCkpSZI0dOhQSd/+wdmyZYt69OihyspKpaWlSfp27h06dEh79+7VE088IUlyOp3q06ePPv/8c895evbsqUsvvVSSNH36dM2ZM0d//vOfNWzYMHXt2lWRkZE6efKkHn74YUVFRam2ttbzx3DAgAGSpM6dO6tv376SpAsvvFANDQ2+fphhsWCa55J0wQUX6K233tLWrVs1ffp0FRUV6dJLL9UVV1wh6dvn6S+//FIS87w16FC1geTkZD377LOeVy8JCQmedUldunRp9r02m63Zx7169dKePXtUX1+vpqYm7d69W9K3bdmPPvpIH3/8sed79+3bpyNHjshmsykiIkLHjx+XJM+5jh07psrKSi1fvlyrV6/W4sWL1djYKJvNJrfbLZvNpscee0z//ve/JUldu3bVZZdd5vnlr6yslCRVVVXJ4XDokksu0SWXXKIVK1aosLBQ9913n5KSkhQfH69f/epXKiws1Jo1azR69GjFxsaqurpaJ0+elPTtEwaCiz/m+Weffaa8vDzPk3afPn3UuXNnhYeHS5LnRcWOHTvUt29fxcfHKykpSYWFhfrjH/+oMWPG6PLLL1efPn20aNEiFRYW6rHHHtNPfvITxcfHe35/jh496uk4bd26VZmZmSooKFBYWJiGDx+u0tJSHT58WEuXLtXDDz+s+vp6zwuV7/8sCG7BMs/z8vL097//XZIUHR2zDprQAAACDUlEQVTtqfXo0aOevx1nztfSz4L/RoeqDQwfPlyzZs1Sfn6+JCkiIkKdO3f2JP5ziYuL04MPPqhJkyYpLi5OnTp1kvTtL8DKlSv15JNPasmSJWpsbFSHDh00Z84cXXbZZRo/frxmzpypN954w3NtvHv37jp+/LhuvfVWRUVFKT09XR06dNC1116rJUuW6Omnn9bTTz+tnJwcNTU1yWazadCgQfrNb36jTZs2qaqqSpMnT1Z1dbVyc3MVHh6uxx9/XFOmTJHb7VZ0dLTy8/OVkJCgWbNm6a677pLD4VBqaqoiIiK0YMECZWRk6MILL1SHDky9YOOPeX7zzTdr7969Gj9+vKKiouR2uzVt2jR17txZkvT666/r+eefV6dOnZSfn68uXbpo27ZtSk1NVW1trX7xi18oJiZGeXl5mj59upqamiRJ8+bNU58+fWS32zV+/Hj17NnT0z3t06ePZs6cqYiICPXr1085OTn66quvtGLFCk2YMEERERHq1auX540cCC3BMs/T0tKUl5en5cuXKywsTHl5eZ6fZ86cOTp8+LCuvfZa3XjjjbwxqZW4OTJaZePGjfriiy/06KOPWl0KIOn//yBceeWVVpcC+E1bz/Pk5GSVlZW1ybmCDZf8AAAADNGhAgAAMESHCgAAwBCBCgAAwBCBCgAAwBCBCgAAwBCBCgAAwBCBCgAAwND/AWp9H1hubc8QAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['WindGustSpeed','WindSpeed9am','WindSpeed3pm']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### All wind speed values also are in sensible ranges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025e88edd8>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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RY0bN67eIwDAHnVoUHXr1i033HBDRx4S7EwB0Ok5sScAQCFBBQBQSFABABQSVAAAhQQVAEAhQQUAUEhQAQAUElQAAIUEFQBAIUEFAFCoQ3/1TC329JubAQAOBpVqtVqt9xAAAAczP/IDACgkqAAACgkqAIBCggoAoJCgAgAoJKioyapVqzJlypT3XPfTn/40S5Ysqel4L730Uu644473XT9lypSsWrUqy5cvzwMPPJAkeeCBB9La2vofj3XjjTfmm9/8ZsaPH5/nn3++pnmA+ujMa8vcuXNz4YUXZvTo0XnhhRdqmodDV93PQwVJcsopp+SUU075j7ePHDmy/fJdd92Vr3/96x94vyeeeCKvvvpqHnroofztb3/Ld7/73ZoXYuDg11Fry4svvpjnnnsuDz74YNavX58rrrgiS5cu7fB5OXgJKjrclClTMnfu3CTJiBEjsmLFikybNi09evTIhg0bsmvXrpx33nl54oknsnHjxsybNy8bN27M/fffn7lz5+bee+/N4sWLc/TRR2fr1q1JkiVLluSVV17Jxz/+8WzevDlTpkzJiSeemGOPPTYXX3xx3n777XznO9/Jueeem7POOivdunVL3759071792zevDmvvvpq+79S33nnncyaNSs9e/bMlClT8pGPfCTr1q3LV77ylbz88st58cUXc/bZZ+eHP/xh3T6HwPvVc21ZsmRJ5s+fn0qlkg0bNuSoo45KkkybNi3VajUbN27MP/7xj8yaNSsNDQ3Wli7Ij/yo2e9+97uMHz++/c+jjz66x/t/9KMfzYIFCzJgwICsW7cud999d770pS/l8ccfb7/P9u3bs3Dhwjz44IOZN2/e+7bfL7zwwhx99NHtW++//vWvkySPPvpozj///Jxyyil58skn09ramtdffz1r165NS0tLXn755cyZMycLFy7MF77whTz22GNJktdffz033XRT7rrrrtx2222ZNm1aFi9enIceeqiDP1vA3uqMa0uS9OjRI3Pnzs1ll12Wr371q+2P7d+/fxYuXJhJkyZlzpw5SawtXZEdKmo2fPjw9n8tJv/7Pod/939PxD9o0KAkSVNTUwYMGNB+edeuXe33eeWVV3LiiSemV69eSZJPfepT//H5+/fvn969e2ft2rV55JFHMm/evPTt2zd//OMfc8kll+Tkk0/OJz/5yfTp0yfHHntsbrrpphx++OHZtGlTTj/99PZjHHHEEenVq1eOOuqo9OnTJ0lSqVRq/bQAhTrj2vJPU6ZMyaWXXpoxY8ZkyJAh7fMmyWc+85ncfPPN7cewtnQtdqjoUFu2bMnmzZuTJOvXr8/bb7/dftveLCT9+/fP2rVr88477+Tdd9/NSy+99L77VCqVtLW1JUlGjx6dn//85zn22GPTt2/fvPrqq+nXr1/uu+++XHrppalUKmlqasqPfvSj3HzzzbnllltyzDHHtC/GFjc4ONR7bXn66afz4x//OEnS0NCQHj16tD/vmjVrkiTPPvtsBg4cuNczcWixQ0WHOvLII3PEEUfkwgsvzCc+8Ykcf/zx+/T4vn375qqrrsrYsWPTt2/fHHbYYe+7z5AhQ/K9730vCxcuzBe/+MXccMMN7dvsxx13XJ588sk89NBDaWhoyPTp05MkX/va1zJ69Og0NTXlqKOOyptvvln+lwUOmHqvLUOHDs1jjz2WsWPHpq2tLRdffHH69++fJFm+fHmWLVuWtra2zJw5s/wvy0HJL0fmoNbS0pJvfetbWbx4cbp1s+EKdIy9XVumTZuW88477z3/tyBdk+9AHLSeffbZjB49OldccYWYAjqMtYVa2KECACgkvQEACgkqAIBCggoAoJCgAgAoJKgAAAoJKgCAQv8PM8ELxWuEs6YAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['Humidity9am','Humidity3pm']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### As we can see that there are some humidity values =0% which is almost never possible, hence removing 0 values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_final= data_final[data_final['Humidity3pm']!=0.0]\n",
    "data_final= data_final[data_final['Humidity9am']!=0.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025e8efcc0>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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gF7PmdIBAAoJKgCAQoIKAKCQoAIAKCSoAAAKCSoAgEKCCgCgUI+eh6oW/j4bALC3q1Sr1Wq9hwAA2Jt5yQ8AoJCgAgAoJKgAAAoJKgCAQoIKAKCQoKImK1euzIwZM95z3c9+9rM88MADNR3vhRdeyI033vi+62fMmJGVK1emtbU1d999d5Lk7rvvTmdn54ce66qrrso3vvGNTJ48Oc8++2xN8wD10ZvXloULF+acc87J+PHj89xzz9U0D/uuup+HCpLkhBNOyAknnPCht48dO7b78i233JKvfe1rH3i/Rx99NC+//HLuu+++/P3vf893vvOdmhdiYO/XU2vL888/n2eeeSb33HNP1qxZk4suuijLli3r8XnZewkqetyMGTOycOHCJMmYMWOyYsWKzJo1K/3798/atWuzbdu2nHnmmXn00Uezbt26LFq0KOvWrctdd92VhQsX5o477si9996bQw89NBs3bkySPPDAA3nppZdy1FFH5c0338yMGTNyzDHH5PDDD895552Xt99+O9/+9rdzxhln5JRTTklDQ0MGDx6cfv365c0338zLL7/c/VvqO++8k3nz5mW//fbLjBkz8tGPfjSvvfZavvzlL+fFF1/M888/n1NPPTU//OEP6/Y1BN6vnmvLAw88kMWLF6dSqWTt2rU55JBDkiSzZs1KtVrNunXr8s9//jPz5s1LY2OjtaUP8pIfNfv973+fyZMnd//30EMP7fD+H/vYx7JkyZIMGzYsr732Wm699dZ86UtfyiOPPNJ9n82bN2fp0qW55557smjRovdtv59zzjk59NBDu7fef/3rXydJHnrooZx11lk54YQT8thjj6WzszOvvvpqVq9enY6Ojrz44otZsGBBli5dmi984Qt5+OGHkySvvvpqrr766txyyy25/vrrM2vWrNx777257777evirBeys3ri2JEn//v2zcOHCfO9738tXvvKV7s8dOnRoli5dmmnTpmXBggVJrC19kR0qajZ69Oju3xaT//c+h//0f0/EP3z48CRJc3Nzhg0b1n1527Zt3fd56aWXcswxx2TAgAFJkk9/+tMf+vhDhw7NwIEDs3r16jz44INZtGhRBg8enD/96U85//zzc/zxx+eTn/xkBg0alMMPPzxXX311DjzwwKxfvz4nnnhi9zEOOuigDBgwIIccckgGDRqUJKlUKrV+WYBCvXFt+ZcZM2bkggsuyIQJEzJixIjueZPks5/9bK655pruY1hb+hY7VPSoDRs25M0330ySrFmzJm+//Xb3bTuzkAwdOjSrV6/OO++8k3fffTcvvPDC++5TqVTS1dWVJBk/fnxuuummHH744Rk8eHBefvnlDBkyJHfeeWcuuOCCVCqVNDc358c//nGuueaaXHvttTnssMO6F2OLG+wd6r22PPnkk/nJT36SJGlsbEz//v27H3fVqlVJkqeffjrHHnvsTs/EvsUOFT3q4IMPzkEHHZRzzjknn/jEJ3LkkUfu0ucPHjw4l1xySSZOnJjBgwfngAMOeN99RowYke9+97tZunRpvvjFL+bKK6/s3mY/4ogj8thjj+W+++5LY2NjZs+enST56le/mvHjx6e5uTmHHHJI3njjjfJ/LLDH1HttGTlyZB5++OFMnDgxXV1dOe+88zJ06NAkSWtra5YvX56urq7MnTu3/B/LXskfR2av1tHRkW9+85u5995709BgwxXoGTu7tsyaNStnnnnme/5vQfomP4HYaz399NMZP358LrroIjEF9BhrC7WwQwUAUEh6AwAUElQAAIUEFQBAIUEFAFBIUAEAFBJUAACF/hd4lfaOXrfItQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['Humidity9am','Humidity3pm']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x20268fc8400>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['Pressure9am','Pressure3pm']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Here also there are no outliers, all pressure ranges also normally can happen in nature "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025edc7fd0>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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XHTRRTn//PNz33335Zvf/OZh3+6mTZvS19eXa6+9NnfccUd+9rOfpbu7O5/+9KezcOHCCfwdAFNRM86We++9N5s3b05bW1uWLVvmbOGQiSTGtWfPnnzxi1/MunXrcuaZZ2b//v35/Oc/n5NPPnnC7uOpp57KT3/609x///1JkkWLFmXBggU5/vjjJ+w+gKmlGWfLiy++mB/84Af58Y9/nKGhoXzgAx/I+973vrS1tU3YfXD0EkmM65e//GXe/e5358wzz0ySTJs2LWvWrMkTTzyRRx99NEmyZcuW3HXXXZk+fXrOPPPM3HzzzXnggQeyffv2fOlLX8rQ0FAuu+yyPPTQQ3n88cdz6623ZubMmWlvb8+5556bbdu25fzzz0+j0UiSnHHGGXnqqacyY8aMrF69OiMjI9mzZ09WrlyZ8847LxdffHHe8Y535K9//WsWLFiQl156KX/4wx8yd+7crF27tlUPFfA/aMbZctJJJ+UnP/lJOjo6smPHjpx44olpa2vLunXrsn379uzevfvA2fLOd77T2cJBvCaJcT3//PN585vffNDHurq60tnZmSTp7+/PunXrctddd2XTpk054YQTct99973m7X3961/PN77xjXzve9/LaaedliR5y1vekscffzwDAwPp7+/PE088kcHBwTzzzDNZsWJF7rzzznzqU5/K5s2bkyQ7duzI9ddfn3vuuSd33313rrrqqtx///3p7e3Nnj17JumRACZSM86WJOno6Mg999yTK664IpdccsmBjx933HG5++67s3bt2tx8881JnC0cTCQxrlNPPTX/+Mc/DvrY3/72tzz22GMHfnz22Wenu7s7SfKud70rTz/99EGf/59v7L5r167MnTs3SXLeeeclSc4666x87GMfy2c+85msWbMmb3/72zN79uyccsopuf3227NixYr8/Oc/z759+5Iks2bNyqmnnprOzs7MmDEjZ599dtra2nLCCSdkaGhoch4IYEI142z5l49//ON5+OGH89hjj+V3v/tdkmTBggVJknPOOSd9fX1JnC0cTCQxroULF+bhhx/Os88+myQZHh7O6tWrM3v27CTJaaedlm3btuWVV15Jkjz66KOZO3duGo1GXnjhhSTJn//85wO3d/LJJ2fbtm1Jkj/+8Y9J/vm6gf7+/mzatClf+cpXsnPnzpxzzjn52te+luuuuy5r1qzJ/PnzDxyIXk8AR75mnC3bt2/P5z73uYyOjqazszPTp09Pe3v7QV/7l7/8JW984xuTOFs4mNckMa7u7u6sXr06K1euzOjoaF5++eUsXLgwZ511Vh5//PGcdNJJufbaa/OJT3wi7e3tOf300w+8VmDTpk258sor87a3vS1dXV1JkrVr12bFihXp6upKV1dXZs6cmdmzZ+e5557Lhz/84XR2dmb58uWZNm1aLr/88nz2s5/NnDlz8qY3vSn9/f0tfjSAidKMs2XevHl561vfmiuuuCJtbW258MILc/755+eRRx7Jk08+mauvvjqDg4O55ZZbWvxoMBX5BrcAHHPWrVuXN7zhDbnyyitbPYUpzD+3AQAUXEkCACi4kgQAUBBJAAAFkQQAUBBJAAAFkQQAUBBJAACF/wN+FZ7lVB/VPQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.boxplot(data=data_final[['Cloud9am','Cloud3pm']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025ee2b8d0>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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D4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgClED4ApRA+AKUQPgCmd/HywWCymJUuW6LffflOXLl20fPly9evXz8+nwG18+umn2rdvX6rHaHMzZsxI9Qht4vHHH1dpaWmqxzDB1+h99913amho0JdffqnDhw9r1apV2rhxo59P4dkbb7yhS5cupXqMNhONRnXt2rVUj9HmLl68mOoR2sQ333zTof/R6t69u957771UjyHJ5+iFw2GNHj1akjRixAhVVlb6+fCtcvLkSdXV1aV6DLRSLBZL9Qhtoq6urkP/fkaj0VSPkOBr9KLRqDIzMxMfB4NB3bhxQ5063f5pwuGwnyPcVjAYVCAQaJfnSoV4PJ7qEdBKHfn3MxgMttv/683xNXqZmZmqra1NfByLxZoMniTl5+f7OcJtlZeXt8vzwH/jx4+XJO3atSvFk+C/oqnA+nr2Ni8vT3v37pUkHT58WA8//LCfDw+jdu3aRfDgG193emPHjtW+ffs0ZcoUxeNxrVixws+HB4BW8zV6aWlpWrp0qZ8PCQC+4uJkAKYQPQCmED0AphA9AKYQPQCmED0AphA9AKYQPQCmED0AphA9AKb4+jI0L1x5uxkANgTivBEbAEM4vAVgCtEDYArRA2AK0QNgCtEDYArRQ7s4ePCgHnvsMYVCIYVCIT377LOaM2eOGhoakn7/pk2bdPTo0ds+3tGjR/X0009rzZo1t32+V155RdL/3UgbuCnl1+nBjlGjRun9999PfPzqq6/q+++/17hx4/71vbNmzWrysSoqKjRlyhSFQiHf50THRvSQEg0NDbpw4YLuueceLVq0SOfOnVN1dbXGjBmjuXPnav78+Xrqqad08eJF/fDDD7p27ZpOnTqlmTNnatCgQdq2bZs6d+6snj17qrGxUV988UXisT/44IMU/mRwHdFDuzlw4IBCoZAuXbqktLQ0Pffcc+rTp49GjBihSZMmqb6+PhG9W0WjUX3yySc6ceKEXnjhBX399deaMGGCevToobFjx+rjjz/Wpk2b1LVrV5WVlamiokL3339/in5KuI7ood3cPLytrq5WaWmpevfurZycHEUiER04cECZmZlJ/8Y3ePBgSdIDDzyQ9Ovdu3fXvHnzlJGRoWPHjmnEiBFt/rPgv4vood3de++9Wr16taZNm6aSkhJlZWVp6dKlOnnypLZs2aJ/vjIyEAjc9rFqamr04Ycfas+ePZKk6dOn/+u/B25F9JASgwYNUigU0i+//KLjx48rHA6ra9eu6tevny5cuNDix8nMzFReXp4mTJig9PR0ZWdn68KFC+rdu3cbTo//Mt5wAIApXKcHwBSiB8AUogfAFKIHwBSiB8AUogfAFKIHwBSiB8CU/wFQSvcaB+0kIgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(5, 6))\n",
    "sns.boxplot(data=data_final[['Rainfall']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2025eddd390>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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l112mdm7d68pKioyTz/9tGltbTUbNmwwu3fvNi+//LK55ZZbTFtbm6mtrTVTp041xhgzefJkc+TIEZOXl2eefPJJY4wxL7zwglm8eLGpr683U6dONYcPHzbGGDN//nyzceNGfzZKN2Lk9hWi7ZZWVFTo1Vdf1b59+3T55ZcrISFBp59+upYvX65+/fqprq5O559/viRp3LhxGjhwoCTp3HPP1Y4dO7R9+3ZdcMEFkqTx48drz549kj47njdo0CBJ0rBhw/TEE0+oT58+CoVCSk5OjlpffX29kpOTdfrpp0uSLrjgAj322GO67LLLNHbsWCUkJCghIUF9+vRxf+Mgpo8//ljJyckqLCyUJFVVVWnevHkaMmRI5HfMcYuDRowYEXmvhw4dqubmZt12221avXq1br75Zp1++uk699xzJUkTJkyQ4zgaOnSojhw58qW+x48fL0k644wz9Pe//107d+7UwYMHNW/ePElSKBRSbW2tN//wHoSzpSfgwgsv1NatW/Xyyy9r+vTpkqRFixZpxYoVKioq0rBhwyIf2O3bt6upqUmtra3617/+pdGjR+vss8/W3/72N0nS1q1bIx/0uLj/fxuWL1+uu+++W8XFxRo7dmzk9RzH+dyXYdCgQWpsbNTevXslSe+++66+8Y1vRH4X/vrwww+1dOlSNTc3S5JGjRql/v37a+DAgdq3b58k6f3334/8frT37PXXX9e0adNUUlKiMWPG6MUXX/zK3z3eF58fPny4zjzzTK1du1YlJSW68cYbdd5553Xp33cyYOT2FTZv3qw5c+Z87rE1a9YoJydHFRUVGjlypCTpmmuu0XXXXaeUlBQNGTIkEjbHjrvt379fV155pcaNG6cFCxZo8eLFWrt2rVpaWrR8+fIv9fv9739fP/nJT3TaaafpjDPOUH19vSTp29/+thYsWKCHHnpI0mcf4Icfflh33XWXHMfRgAEDVFhYqH//+99ebhZ00BVXXKHt27drxowZ6tevn4wxWrBggXr16qUHH3xQZ555poYNG9bua6Snpys/P1/9+vWL/H9btmw54VoGDx6sW265RXPmzFFra6u+/vWva+rUqZ39p500WDjvgV27dumnP/1p5C8tgO7HbikAKzFyA2AlRm4ArES4AbAS4QbASoQbACsRbgCsRLgBsNL/AdpqmWs7NDBUAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 360x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "plt.figure(figsize=(5, 6))\n",
    "sns.boxplot(data=data_final[['Evaporation','Sunshine']])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Hence  the data is not suffering from outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2>2. Data pre-processing and Feature engineering</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "From the EDA we observed that there are few categorical variables they are :\n",
    "\n",
    "WindGustDir\n",
    "\n",
    "RainToday\n",
    "\n",
    "WindDir9am\n",
    "\n",
    "WindDir3pm\n",
    "\n",
    "and \n",
    "class label = RainTomorrow\n",
    "\n",
    "so encoding all these features into values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "WindGustDir_encode = LabelEncoder()\n",
    "data_final['WindGustDir']=WindGustDir_encode.fit_transform(data_final['WindGustDir'])\n",
    "\n",
    "WindDir9am_encode = LabelEncoder()\n",
    "data_final['WindDir9am']=WindDir9am_encode.fit_transform(data_final['WindDir9am'])\n",
    "\n",
    "WindDir3pm_encode = LabelEncoder()\n",
    "data_final['WindDir3pm']=WindDir3pm_encode.fit_transform(data_final['WindDir3pm'])\n",
    "\n",
    "RainToday_encode = LabelEncoder()\n",
    "data_final['RainToday']=RainToday_encode.fit_transform(data_final['RainToday'])\n",
    "\n",
    "RainTomorrow_encode = LabelEncoder()\n",
    "data_final['RainTomorrow']=RainTomorrow_encode.fit_transform(data_final[\"RainTomorrow\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y= data_final['RainTomorrow']\n",
    "X = data_final.drop(['RainTomorrow'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "column_names=X.columns.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.80,shuffle=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Since the features are measured from different scales, appling featruring scaling  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler= StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plotErrors(k,train,cv):\n",
    "       \n",
    "    plt.plot(k, train, label='Train logloss')\n",
    "    plt.plot(k, cv, label='CV logloss')\n",
    "    plt.legend()\n",
    "    plt.xlabel(\"log(C)= -log(λ)\")\n",
    "    plt.ylabel(\"Neg_Log Loss\")\n",
    "    plt.title(\"Error Plot for Train and Validation data\")\n",
    "    plt.grid()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h2> 3. Modeling the data using Logisitic Regression</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Hyper-parameter tuning</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "parameters={'C':[10**-6,10**-5,10**-4, 10**-2, 10**0, 10**2, 10**3] }\n",
    "log_c = list(map(lambda x : float(math.log(x)),parameters['C']))\n",
    "\n",
    "clf_log = LogisticRegression(penalty='l2',class_weight='balanced')\n",
    "\n",
    "clf = GridSearchCV(clf_log, parameters, cv=5, scoring='neg_log_loss',return_train_score =True)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "train_loss= clf.cv_results_['mean_train_score']\n",
    "cv_loss = clf.cv_results_['mean_test_score'] \n",
    "\n",
    "plotErrors(k=log_c,train=train_loss,cv=cv_loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1000, class_weight='balanced', dual=False,\n",
       "          fit_intercept=True, intercept_scaling=1, max_iter=100,\n",
       "          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,\n",
       "          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = clf.best_estimator_\n",
    "clf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1000, class_weight='balanced', dual=False,\n",
       "          fit_intercept=True, intercept_scaling=1, max_iter=100,\n",
       "          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,\n",
       "          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Trainig with the best value of C\n",
    "clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Model Evaluating</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Log_loss on train data is :0.4213743265965361\n",
      "Log_loss on test data is :0.42842652530959185\n"
     ]
    }
   ],
   "source": [
    "#Printing the log-loss for both trian and test data\n",
    "train_loss = log_loss(y_train, clf.predict_proba(X_train)[:,1])\n",
    "test_loss  =log_loss(y_test, clf.predict_proba(X_test)[:,1])\n",
    "\n",
    "\n",
    "print(\"Log_loss on train data is :{}\".format(train_loss))\n",
    "print(\"Log_loss on test data is :{}\".format(test_loss))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Since log-loss can have any value between [0, ∞] so we can only interpret the model prefectly fitted or not but   we cannot tell how best the model is, hence checking with AUC metric "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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1jy3M5vN6PV6dDoddrv9Lx07KyuL/fv3s3DhQgD8/f3p0qUL27dvp169ehgMBurXr/+X9l2/fn0MBsefXb9+/di3bx+LFi3izJkznDx5knr16t20Tbly5ahZsyYADz30EF9++eVN6zRo0IBixYqxbNkyEhIS2LNnD76+vs7lzZs3d54Bh4WFkZqaSkxMDGFhYVSrVg2A7t278/bbb9/1Z/rxxx/59ddfWb58OQA5OTkYjUb69OlDaGgoXbp0oUWLFrRo0YJHH330lvuKjo6mXbt2zti7du3KihUrAHjttdfYtWsX8+fP58yZMyQlJZGVlXXTPu50vcjISHbs2EFISAjNmjXj4MGDnD59mq1bt9KuXbvbfu5OnToBULNmTXJycsjMzMTf3/+m9Xr06AFAjRo1qFy5MocOHSI6OprffvuN7t27O9dLSUkhPT2dJ598kvHjx7NlyxaaNGni7MMX954kgCKuVq1ajB49mlGjRlGzZk0qVKhAREQEn3zyCXa7HZ3uj/v4drudvXv38vzzz1OtWjWsVitnzpyhcuXKznVMJhMvvfQSkydPpnTp0nke0263Oy+/b5x3PekYjUZnI363fHx8nD/PnDmTQ4cO0bVrVx555BGsVutNxwXw8vJy/qxpWp7r/PDDD0yZMoV///vfREZGUrVqVdatW3fbfdy4r7/6maxWK++//77ze05NTUWn02EwGPj88885dOgQP/30E5MnT6ZNmzYMHz78lvvLL6Zhw4ahaRpPPPEEbdq04dy5c3l+F3e6Xtu2bfnwww+pWrUqTZo0wc/Pjx07drBz505Gjhx52899PTZN026K+0Z6vd75s91ux2AwYLPZ6Nq1K6+88grguOF85coV/P396d27N48//jg7d+5k+/btvP/++2zatOm28Yi7J08B3Qc6duxI3bp1mTZtGgDt2rXD29vb2V8LjrPOSZMm4evrS9u2bTEajTz33HO8/vrrXL16FXBcHUydOpXs7Ox8G38APz8/6tWrx9KlSwFIT09n7dq1NGnS5J5+rp07d9KvXz86d+5McHAw0dHR2Gy2v7SvXbt20bp1a3r16kXt2rXZsmXLbff18MMPExcXR2xsLABr1qz5S8du1qwZixcvRimFyWRi8ODBfPHFFxw9epSnnnqK6tWrM2TIEPr27et8kstgMGCxWG7aV4sWLfjmm29IT0/H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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Plotting AUC \n",
    "train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(X_train)[:,1])\n",
    "test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])\n",
    "plt.plot(train_fpr, train_tpr, label=\"train AUC =\"+str(auc(train_fpr, train_tpr)))\n",
    "plt.plot(test_fpr, test_tpr, label=\"test AUC =\"+str(auc(test_fpr, test_tpr)))\n",
    "plt.legend()\n",
    "plt.xlabel(\"FPR\")\n",
    "plt.ylabel(\"TPR\")\n",
    "plt.title(\"ROC for Train and Test data with best_fit\")\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#ploting Confusion matrix\n",
    "#Ref:https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785\n",
    "def plotcm (cm,ax,title):\n",
    "    sns.heatmap(cm, ax=ax,annot=True,fmt='d',annot_kws={'size':20},yticklabels=3);\n",
    "    ax.set_xlabel('Predicted');\n",
    "    ax.set_ylabel('Actual'); \n",
    "    ax.set_title('Confusion Matrix for {} '.format(title)); \n",
    "    ax.xaxis.set_ticklabels(['NO', 'YES']); \n",
    "    ax.yaxis.set_ticklabels(['NO', 'YES']);\n",
    "\n",
    "\n",
    "fig, subplt = plt.subplots(1, 2,figsize=(10, 4)) \n",
    "cm=confusion_matrix(y_train, clf.predict(X_train))\n",
    "ax= subplt[0]\n",
    "plotcm(cm,ax,'Train data')\n",
    "\n",
    "cm=confusion_matrix(y_test, clf.predict(X_test))\n",
    "ax= subplt[1]\n",
    "plotcm(cm,ax,'Test data')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Feature interpretation</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.27323143, -0.02141824,  0.10994747, -0.06853409, -0.62721434,\n",
       "        0.12271816,  0.85054113, -0.0318672 , -0.03485931, -0.09530709,\n",
       "       -0.24165585,  0.06371587,  1.13844823,  1.00665528, -1.42428327,\n",
       "       -0.09042131,  0.30186218,  0.2647275 ,  0.13053517,  0.17729874])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.coef_[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_weights=sorted(zip(clf.coef_[0],column_names),reverse = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(1.1384482301413, 'Humidity3pm'),\n",
       " (1.0066552763622452, 'Pressure9am'),\n",
       " (0.8505411317523849, 'WindGustSpeed'),\n",
       " (0.30186217790397973, 'Cloud3pm'),\n",
       " (0.26472749783217453, 'Temp9am'),\n",
       " (0.17729874034043772, 'RainToday'),\n",
       " (0.130535165126773, 'Temp3pm'),\n",
       " (0.12271816167719675, 'WindGustDir'),\n",
       " (0.10994746540819095, 'Rainfall'),\n",
       " (0.06371587289699299, 'Humidity9am'),\n",
       " (-0.021418239382843207, 'MaxTemp'),\n",
       " (-0.0318671999898255, 'WindDir9am'),\n",
       " (-0.034859307001313636, 'WindDir3pm'),\n",
       " (-0.06853408644974304, 'Evaporation'),\n",
       " (-0.09042130532728501, 'Cloud9am'),\n",
       " (-0.0953070902750138, 'WindSpeed9am'),\n",
       " (-0.24165584539227303, 'WindSpeed3pm'),\n",
       " (-0.27323142640391773, 'MinTemp'),\n",
       " (-0.6272143358588156, 'Sunshine'),\n",
       " (-1.4242832719023153, 'Pressure3pm')]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature_weights"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is our weight vector, each weight corresponds to each feature, above are the sorted features based on its weight value.\n",
    "Higher weight value means higher important feature it is.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "def will_rain_fall_for_this_conditions(xq):\n",
    "    \n",
    "    xq[\"WindGustDir\"]=WindGustDir_encode.transform([xq[\"WindGustDir\"]])\n",
    "    xq[\"WindDir9am\"]=WindDir9am_encode.transform([xq[\"WindDir9am\"]])\n",
    "    xq[\"WindDir3pm\"]=WindDir3pm_encode.transform([xq[\"WindDir3pm\"]])\n",
    "    xq[\"RainToday\"]=RainToday_encode.transform([xq[\"RainToday\"]])\n",
    "    xq=np.array(list((xq.values())))\n",
    "    final_xq = scaler.transform(xq.reshape(1, -1))\n",
    "    chance=clf.predict_proba(final_xq)[:,1]\n",
    "    if chance>=0.5:\n",
    "        print(\"Yes, there is a {} % chance of rain can fall on tommorow \".format(chance*100))\n",
    "    else:\n",
    "        print(\"No, there is only {}% chance of rainfall hence we cannot expect rain on tommorow \".format(chance*100))\n",
    "    print(\"Because today's Humidity at 3pm ={}%,Atmosphereic Pressure at 9am={}millibars,and Wind Gust Speed ={}km/hr, which are very good sign for rainfall\"\n",
    "          .format(Humidity3pm,Pressure9am,WindGustSpeed)) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Giving one query point here\n",
    "\n",
    "MinTemp   = 26.2\n",
    "MaxTemp   = 31.7\n",
    "Rainfall   = 2.8\n",
    "Evaporation   = 5.4\n",
    "Sunshine   = 3.5\n",
    "WindGustDir   = \"NNW\"\n",
    "WindGustSpeed   = 57\n",
    "WindDir9am   = \"NNW\"\n",
    "WindDir3pm   = \"NNW\"\n",
    "WindSpeed9am   = 20\n",
    "WindSpeed3pm   = 13\n",
    "Humidity9am   = 81\n",
    "Humidity3pm   = 95\n",
    "Pressure9am   = 1007.2\n",
    "Pressure3pm   = 1006.1\n",
    "Cloud9am   = 7\n",
    "Cloud3pm   = 8\n",
    "Temp9am   = 28.8\n",
    "Temp3pm   = 25.4\n",
    "RainToday   =\"Yes\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "point = [MinTemp,MaxTemp,Rainfall,\n",
    "         Evaporation,Sunshine,WindGustDir,\n",
    "         WindGustSpeed,WindDir9am,WindDir3pm,\n",
    "         WindSpeed9am,WindSpeed3pm,Humidity9am,\n",
    "         Humidity3pm,Pressure9am,Pressure3pm,\n",
    "         Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday]\n",
    "\n",
    "xq=dict()\n",
    "for i,name in enumerate(column_names):\n",
    "    xq[name]=point[i]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Interpretting the Classifer result</h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Yes, there is a [99.04229122] % chance of rain can fall on tommorow \n",
      "Because today's Humidity at 3pm =95%,Atmosphereic Pressure at 9am=1007.2millibars,and Wind Gust Speed =57km/hr, which are very good sign for rainfall\n"
     ]
    }
   ],
   "source": [
    "will_rain_fall_for_this_conditions(xq)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
