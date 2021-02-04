import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 


def preprocessor():

	train_df = pd.read_csv('./Data/train.csv')
	test_df = pd.read_csv('./Data/test.csv')



	


	def feature_engineering(df):
    
    	#Combine the SF for outdoor area
    	df['Total_OutdoorSF'] = df['3SsnPorch']+df['EnclosedPorch']+df['OpenPorchSF']+df['ScreenPorch']+df['WoodDeckSF']
    	df.drop("OpenPorchSF", axis = 1, inplace = True)
    	df.drop("EnclosedPorch", axis = 1, inplace = True)
    	df.drop("3SsnPorch", axis = 1, inplace = True)
    	df.drop("ScreenPorch", axis = 1, inplace = True)
    	df.drop("WoodDeckSF", axis = 1, inplace = True)
    
    	#sum all bathrooms into new column 'Baths'
    	df['Total_Baths'] = df['BsmtHalfBath'] + df['BsmtFullBath'] + df['HalfBath'] + df['FullBath']
    	df['Total_Baths'].fillna(df.Total_Baths.mode()[0], inplace=True)
    	df.drop("BsmtHalfBath", axis = 1, inplace = True)
    	df.drop("BsmtFullBath", axis = 1, inplace = True)
    	df.drop("HalfBath", axis = 1, inplace = True)
    	df.drop("FullBath", axis = 1, inplace = True)

    
    	#Change years to ages (note that 53% of houses have same year for YearBuilt and YearRemodAdd):
    	#Change House_Age to Age (YrSold - YearBuilt)
    	df['House_Age'] = df['YrSold'] - df['YearBuilt']

    	#Change YearRemodAdd to House_Age_Remod(YrSold - YearRemodAdd) 
    	df['House_Age_Remod'] = df['YrSold'] - df['YearRemodAdd']
    	df.drop(['YearBuilt'], axis=1, inplace=True)
    	df.drop(['YearRemodAdd'], axis=1, inplace=True)
    
    	#Change GarageYrBlt to Garage_Age (YrSold - GarageYrBlt)
    	df['Garage_Age'] = df['YrSold'] - df['GarageYrBlt']
    	df.drop(['GarageYrBlt'], axis=1, inplace=True)
    
    	df.drop(['Utilities'], axis=1, inplace=True)
    	df.drop(['Condition2'], axis=1, inplace=True)
    	df.drop(['BsmtCond'], axis=1, inplace=True)
    	df.drop(['BsmtExposure'], axis=1, inplace=True)
    	df.drop(['BsmtFinType1'], axis=1, inplace=True)
    	df.drop(['BsmtFinSF1'], axis=1, inplace=True)
    	df.drop(['BsmtFinType2'], axis=1, inplace=True)
    	df.drop(['BsmtFinSF2'], axis=1, inplace=True)
    	df.drop(['BsmtUnfSF'], axis=1, inplace=True)
    	df.drop(['Heating'], axis=1, inplace=True)
    	df.drop(['1stFlrSF'], axis=1, inplace=True)
    	df.drop(['2ndFlrSF'], axis=1, inplace=True)
    	df.drop(['LowQualFinSF'], axis=1, inplace=True)
    	df.drop(['KitchenAbvGr'], axis=1, inplace=True)
    	df.drop(['Functional'], axis=1, inplace=True)
    	df.drop(['Fireplaces'], axis=1, inplace=True)
    	df.drop(['MiscVal'], axis=1, inplace=True)
    	df.drop(['Street'], axis=1, inplace=True)
    	df.drop(['Alley'], axis=1, inplace=True)
    	df.drop(['RoofMatl'], axis=1, inplace=True)
    	df.drop(['Fence'], axis=1, inplace=True)
    	df.drop(['LandSlope'], axis=1, inplace=True)
    	df.drop(['MiscFeature'], axis=1, inplace=True)
    
# Features to potentially drop since correlation < ~0.30 and with > ~0.9 (except for MSSubClass)  
    	df.drop(['Electrical'], axis=1, inplace=True)
    	df.drop(['PavedDrive'], axis=1, inplace=True)
    	df.drop(['BedroomAbvGr'], axis=1, inplace=True)
    	df.drop(['PoolQC'], axis=1, inplace=True)
    	df.drop(['ExterCond'], axis=1, inplace=True)
    	df.drop(['OverallCond'], axis=1, inplace=True)
#       df.drop(['AgeRemodAdd'], axis=1, inplace=True)
#       df.drop(['AgeGarage'], axis=1, inplace=True)
#       df.drop(['Age'], axis=1, inplace=True)

    
    return df


	def dummification(df1, df2):
		#split categorical and numerical variables to dummify categorical varialbes (concat numerical after dummification)
		train1 = new_train_df.select_dtypes(["object","category"])
		train2 = new_train_df.select_dtypes(["float64","int64"])

		#OneHotEncoder function to dummify
		encoder = OneHotEncoder(categories = "auto",drop = 'first',sparse = False)
		train1_enc = encoder.fit_transform(train1)
		column = encoder.get_feature_names(train1.columns.tolist())

		# Combine the object and numeric features back again for train set
		train_df =  pd.DataFrame(train1_enc, columns= column)
		train_df.set_index(train2.index, inplace = True)
		train_complete = pd.concat([train_df, train2], axis = 1)
