import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import namedtuple 


def preprocessor():

	# Declaring namedtuple
	out_df = namedtuple("out_df", ["train_undum_df", "test_undum_df", "train_dum_df", "test_dum_df"])

	train_df = pd.read_csv('./Data/train.csv')
	test_df = pd.read_csv('./Data/test.csv')

	## Imputation on missing values ########################################
	## Imputation on missing values ########################################
	## Imputation on missing values ########################################

	# Train Set
	#LotFrontage with the mean of each Neighborhood in the train set
	neighbor_mean = dict(train_df.groupby('Neighborhood')["LotFrontage"].mean()) # RK edit 28dec2020: added
	train_df["LotFrontage"] = train_df["LotFrontage"].fillna(train_df["Neighborhood"].map(neighbor_mean)) # RK edit 28dec2020: added

	train_df['Alley'] = train_df['Alley'].fillna('NaN') # RK edit 28dec2020: Added, but not needed, to be droped
	train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
	train_df['MasVnrType'] = train_df['MasVnrType'].fillna('None')
	train_df['BsmtQual'] = train_df['BsmtQual'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['BsmtCond'] = train_df['BsmtCond'].fillna('None') # RK edit 28dec2020: not needed, to be droped
	train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna('None') # RK edit 28dec2020: not needed, to be droped
	train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('None') # RK edit 28dec2020: not needed, to be droped
	train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('None') # RK edit 28dec2020: not needed, to be droped
	train_df['Electrical'] = train_df['Electrical'].fillna("SBrkr") # RK edit 28dec2020: Added, but not needed, to be droped
	train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['GarageType'] = train_df['GarageType'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['YrSold']) # RK edit 28dec2020: Added. Impute to YrSold for Houses without garages (needed for subsequent calculation of AgeGarage)
	train_df['GarageFinish'] = train_df['GarageFinish'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['GarageQual'] = train_df['GarageQual'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['GarageCond'] = train_df['GarageCond'].fillna('None') # RK edit 28dec2020: not needed, to be droped
	train_df['PoolQC'] = train_df['PoolQC'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['Fence'] = train_df['Fence'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	train_df['MiscFeature'] = train_df['MiscFeature'].fillna('None') # RK edit 28dec2020: Added. but not needed, to be droped

	# Test Set
	test_df["MSZoning"].fillna(train_df["MSZoning"].mode()[0], inplace = True)
	#LotFrontage with the mean of each Neighborhood in the test set
	neighbor_mean = dict(train_df.groupby('Neighborhood')["LotFrontage"].mean())
	test_df["LotFrontage"] = test_df["LotFrontage"].fillna(test_df["Neighborhood"].map(neighbor_mean))
	test_df['Alley'] = test_df['Alley'].fillna('NaN') # RK edit 28dec2020: Added, but not needed, to be dropped

	test_df['Utilities'] = test_df['Utilities'].fillna('NaN') # RK edit 28dec2020: Added, but not needed, to be dropped
	test_df["Exterior1st"].fillna(train_df["Exterior1st"].mode()[0], inplace = True)
	test_df["Exterior2nd"].fillna(train_df["Exterior2nd"].mode()[0], inplace = True)

	test_df["MasVnrType"].fillna("None", inplace = True)
	test_df["MasVnrArea"].fillna(0.0, inplace = True) 

	test_df['BsmtQual'] = test_df['BsmtQual'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['BsmtCond'] = test_df['BsmtCond'].fillna('None') # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna('None') # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna('None') # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(0) # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna('None') # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(0) # RK edit 28dec2020: not needed, to be dropped
	test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(0) # RK edit 28dec2020: not needed, to be dropped
	test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(0) # RK edit 28dec2020: Added, for houses without basements
	test_df["BsmtHalfBath"].fillna(train_df["BsmtHalfBath"].mode()[0], inplace = True) # RK edit 28dec2020: not needed, to be droped
	test_df["BsmtFullBath"].fillna(train_df["BsmtFullBath"].mode()[0], inplace = True) # RK edit 28dec2020: not needed, to be droped
	test_df["KitchenQual"].fillna(train_df["KitchenQual"].mode()[0], inplace = True)
	test_df["Functional"].fillna(train_df["Functional"].mode()[0], inplace = True)
	test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('NA') # RK edit 28dec2020: Added

	test_df['GarageType'] = test_df['GarageType'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['YrSold']) # RK edit 28dec2020: Added. Impute to YrSold for Houses without garages (needed for subsequent calculation of AgeGarage)
	test_df['GarageFinish'] = test_df['GarageFinish'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['GarageCars'] = test_df['GarageCars'].fillna(0) # RK edit 28dec2020: house without garage
	test_df['GarageArea'] = test_df['GarageArea'].fillna('None') # RK edit 28dec2020: not needed, to be dropped
	test_df['GarageQual'] = test_df['GarageQual'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['GarageCond'] = test_df['GarageCond'].fillna('None') # RK edit 28dec2020: not needed, to be dropped

	test_df['PoolQC'] = test_df['PoolQC'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['Fence'] = test_df['Fence'].fillna('NA') # RK edit 28dec2020: needs to be 'NA' for ordinal encoding below to work
	test_df['MiscFeature'] = test_df['MiscFeature'].fillna('None') # RK edit 28dec2020: Added. but not needed, to be droped

	test_df["SaleType"].fillna(train_df["SaleType"].mode()[0], inplace = True)


	## Feature Engineering ########################################
	## Feature Engineering ########################################
	## Feature Engineering ########################################

	def feature_engineering(df):

		#Combine the SF for outdoor area
		df['Total_OutdoorSF'] = df['3SsnPorch']+df['EnclosedPorch']+df['OpenPorchSF']+df['ScreenPorch']+df['WoodDeckSF']
		df.drop("OpenPorchSF", axis = 1, inplace = True)
		df.drop("EnclosedPorch", axis = 1, inplace = True)
		df.drop("3SsnPorch", axis = 1, inplace = True)
		df.drop("ScreenPorch", axis = 1, inplace = True)
		df.drop("WoodDeckSF", axis = 1, inplace = True)
		
		#sum all bathrooms into new column 'Baths'
		df['Baths'] = df['BsmtHalfBath'] + df['BsmtFullBath'] + df['HalfBath'] + df['FullBath']
		df['Baths'].fillna(df.Baths.mode()[0], inplace=True)
		df.drop("BsmtHalfBath", axis = 1, inplace = True)
		df.drop("BsmtFullBath", axis = 1, inplace = True)
		df.drop("HalfBath", axis = 1, inplace = True)
		df.drop("FullBath", axis = 1, inplace = True)
		
		#Change years to ages (note that 53% of houses have same year for YearBuilt and YearRemodAdd):
		#Change YearBuilt to Age (YrSold - YearBuilt)
		df['Age'] = df['YrSold'] - df['YearBuilt']

		#Change YearRemodAdd to AgeRemodAdd (YrSold - YearRemodAdd) 
		df['AgeRemodAdd'] = df['YrSold'] - df['YearRemodAdd']
		df.drop(['YearBuilt'], axis=1, inplace=True)
		df.drop(['YearRemodAdd'], axis=1, inplace=True)
		
		#Change GarageYrBlt to AgeGarage (YrSold - GarageYrBlt)
		df['AgeGarage'] = df['YrSold'] - df['GarageYrBlt']
		df.drop(['GarageYrBlt'], axis=1, inplace=True)
		
		###############################################
		# Encode ordinals # RK edit 28dec2020: added
		vals = {'Ex' : 5 , 'Gd' : 4, 'TA' : 3 , 'Fa' : 2, 'Po' : 1, 'NA' : 0}
		df['ExterQual'] = df['ExterQual'].map(vals)
		df['ExterCond'] = df['ExterCond'].map(vals)
		df['BsmtQual'] = df['BsmtQual'].map(vals)
		df['KitchenQual'] = df['KitchenQual'].map(vals)
		df['FireplaceQu'] = df['FireplaceQu'].map(vals)
		df['GarageQual'] = df['GarageQual'].map(vals) 
		
		vals2 = {'Ex' : 5, 'Gd' : 4 , 'TA' : 3, 'Fa' : 2, 'Po' : 2} # only 3 'Poor', merge
		df['HeatingQC'] = df['HeatingQC'].map(vals2)
		
		vals3 = {'SBrkr' : 2, 'FuseA' : 1, 'FuseF': 1, 'Mix': 1, 'FuseP': 1} # merge infrequent levels
		df['Electrical'] = df['Electrical'].map(vals3)
		
		vals4 = {'Fin' : 3 , 'RFn' : 2, 'Unf' : 1 , 'NA' : 0} 
		df['GarageFinish'] = df['GarageFinish'].map(vals4)
		
		vals5 = {'Y' : 3 , 'P' : 2, 'N' : 1 }
		df['PavedDrive'] = df['PavedDrive'].map(vals5)
		
		vals6 = {'Y' : 1, 'N' : 0} # change to binary
		df['CentralAir'] = df['CentralAir'].map(vals6)
		
		vals7 = {'GdPrv' : 1 , 'GdWo' : 1, 'MnPrv' : 1 , 'MnWw' : 1, 'NA' : 0} # change to binary
		df['Fence'] = df['Fence'].map(vals7)
		###############################################
		
		# MSSubClass is numeric but actually should be nominal 'object' type as needs to be dummified # RK edit 28dec2020: added
		df['MSSubClass'] = df['MSSubClass'].astype('object')
		
		
		#Remove
		df.drop(['Id'], axis=1, inplace=True) # RK edit 28dec2020: added
		df.drop(['Street'], axis=1, inplace=True)
		df.drop(['Alley'], axis=1, inplace=True)
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
		df.drop(['GarageArea'], axis=1, inplace=True) # RK edit 28dec2020: added
		df.drop(['PoolArea'], axis=1, inplace=True) # RK edit 28dec2020: added
		df.drop(['MiscVal'], axis=1, inplace=True)
		df.drop(['RoofMatl'], axis=1, inplace=True)
		df.drop(['Fence'], axis=1, inplace=True)
		df.drop(['LandSlope'], axis=1, inplace=True)
		df.drop(['MiscFeature'], axis=1, inplace=True)
		df.drop(['MoSold'], axis=1, inplace=True) # RK edit 28dec2020: added
		df.drop(['YrSold'], axis=1, inplace=True) # RK edit 28dec2020: added    
		
	# Features to potentially drop since correlation < ~0.30 and with > ~0.9 (except for MSSubClass)  
		df.drop(['Electrical'], axis=1, inplace=True)
		df.drop(['PavedDrive'], axis=1, inplace=True)
		df.drop(['BedroomAbvGr'], axis=1, inplace=True)
		df.drop(['PoolQC'], axis=1, inplace=True)
		df.drop(['ExterCond'], axis=1, inplace=True)
		df.drop(['OverallCond'], axis=1, inplace=True)
	#     df.drop(['AgeRemodAdd'], axis=1, inplace=True)
	#     df.drop(['AgeGarage'], axis=1, inplace=True)
	#     df.drop(['Age'], axis=1, inplace=True)
		
		return df


	## Apply Feature Engineering Function ##############################################################################
	## Apply Feature Engineering Function ##############################################################################
	## Apply Feature Engineering Function ##############################################################################

	print("Number of features in train set before feature engineering: " + str(train_df.shape[1]))
	print("-"*60)
	new_train_df = feature_engineering(train_df)
	print("Number of features in train set after feature engineering: " + str(new_train_df.shape[1]))
	print("*"*60)
	print("Number of features in test set before feature engineering: " + str(test_df.shape[1]))
	print("-"*60)
	new_test_df = feature_engineering(test_df)
	print("Number of features in test set after feature engineering: " + str(new_test_df.shape[1]))
	print("*"*60)


	## Dummify and Label Encoding of Un-dummified data #################################
	## Dummify and Label Encoding of Un-dummified data #################################
	## Dummify and Label Encoding of Un-dummified data #################################

	#split categorical and numerical variables to dummify categorical varialbes (concat numerical after dummification)
	train1 = new_train_df.select_dtypes(["object","category"])
	train2 = new_train_df.select_dtypes(["float64","int64"])

	#OneHotEncoder function to dummify
	encoder = OneHotEncoder(categories = "auto",drop = 'first' ,sparse = False)
	train1_enc = encoder.fit_transform(train1)
	column = encoder.get_feature_names(train1.columns.tolist())

	#LabelEncoder function to encode undummified version
	le = LabelEncoder()
	train1_le = train1.apply(le.fit_transform, axis = 0)
	column_le = train1.columns.tolist()

	# Combine the object and numeric features back again for train set
	train_df =  pd.DataFrame(train1_enc, columns= column)
	train_df.set_index(train2.index, inplace = True)
	train_complete = pd.concat([train_df, train2], axis = 1)

	# same, but for undummified version of data
	train_le_df =  pd.DataFrame(train1_le, columns= column_le)
	train_le_df.set_index(train2.index, inplace = True)
	train_le_complete = pd.concat([train_le_df, train2], axis = 1)

	#also do this for test set
	#split categorical and numerical variables to dummify categorical varialbes (concat numerical after dummification)
	test1 = new_test_df.select_dtypes(["object","category"])
	test2 = new_test_df.select_dtypes(["float64","int64"]) 
	#OneHotEncoder function to dummify
	encoder = OneHotEncoder(categories = "auto",drop = 'first' ,sparse = False)
	test1_enc = encoder.fit_transform(test1)
	column = encoder.get_feature_names(test1.columns.tolist())

	#LabelEncoder function to encode undummified version
	le = LabelEncoder()
	test1_le = test1.apply(le.fit_transform, axis = 0)
	column_le = test1.columns.tolist()

	# Combine the object and numeric features back again for test set
	test_df =  pd.DataFrame(test1_enc, columns= column)
	test_df.set_index(test2.index, inplace = True)
	test_complete = pd.concat([test_df, test2], axis = 1)

	# same, but for undummified version of data
	test_le_df =  pd.DataFrame(test1_le, columns= column_le)
	test_le_df.set_index(test2.index, inplace = True)
	test_le_complete = pd.concat([test_le_df, test2], axis = 1)

	# Note that test dataframe has 4 fewer columns than train as the following levels 
	# were not present in the test and thus no column was added during dummification
	# add 4 additional orphan columns of zeros to test_complete to make identical to train_complete
	cols = list(set(train_complete.columns.tolist()) - set(test_complete.columns.tolist()))
	vals = [0] * 4
	test_complete = test_complete.reindex(columns = test_complete.columns.tolist() + cols[:-1])   
	test_complete[cols[:-1]] = vals 
	
	print ("Undummified df objects: train_undum_df, test_undum_df")
	print ("Dummified, label encoded df objects: train_dum_df, test_dum_df")

	## Return all four dataframes in namedtuple ##############################################################################
	## Return all four dataframes in namedtuple ##############################################################################
	## Return all four dataframes in namedtuple ##############################################################################
	return out_df(train_le_complete, test_le_complete, train_complete, test_complete)

