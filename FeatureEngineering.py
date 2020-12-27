import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def feature_engineering(df):
	
	










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