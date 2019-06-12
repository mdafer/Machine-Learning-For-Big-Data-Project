#######################
#  ML for BD Project  #
#   Moustafa Dafer    #
#     B00 772009      #
#######################

###Initialization & Imports###
import pandas as pandas
import numpy as np
from dateutil.parser import parse
import os.path
# Seaborn for plotting and styling
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from bearingCalc import calculate_bearing, haversine

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind



def dayGroup(group):
	#discard subtrajectories where points < 10
	if len(group['latitude'])<10:
		return pandas.DataFrame()
	distance=[]
	#calculate distance
	for i in range(len(group['latitude'])-1):
		distance.append(haversine(group['longitude'].iloc[i], group['latitude'].iloc[i], group['longitude'].iloc[i+1],group['latitude'].iloc[i+1]))
	#calculate timediff
	timediff= []
	for i in range(len(group['parsed_time'])-1):
		timediff.append((parse(group['parsed_time'].iloc[i+1]) - parse(group['parsed_time'].iloc[i])).seconds)
	#calculate speed
	speed = np.array(distance, dtype=np.float)/np.array(timediff, dtype=np.float)
	#calculate distributed acceleration
	acceleration = []
	for i in range(len(speed)-1):
		acceleration.append((speed[i+1] - speed[i])/timediff[i+1])
	#uncomment for overall non-distributed acceleration
	#print (speed[-1] - speed[0])/(parse(group.parsed_time.iloc[-1]) - parse(group.parsed_time.iloc[0])).seconds

	#calculate distributed acceleration
	bearing = []
	for i in range(len(group['latitude'])-1):
		bearing.append(calculate_bearing((group['latitude'].iloc[i], group['longitude'].iloc[i]),(group['latitude'].iloc[i+1], group['longitude'].iloc[i+1])))
	return pandas.DataFrame({'t_user_id':group['t_user_id'].iloc[0],'parsed_time':group['parsed_time'].iloc[0],'parsed_date':group['parsed_date'].iloc[0],
		'transportation_mode':group['transportation_mode'].iloc[0],'distance': sum(distance),
		'timediff':sum(timediff),'speed':speed.mean(), 'acceleration': np.mean(acceleration),'bearing':np.mean(bearing),'distance_std':np.std(distance), 'speed_std':np.std(speed), 'acc_std':np.std(acceleration), 'bearing_std':np.std(bearing)}, index=[0])

def question1(userDayTransportDF):
	print userDayTransportDF
	#print userDayTransportDF.columns
	#to avoid calculating points from different transportation modes
	userDistanceDF = userDayTransportDF.groupby(['t_user_id','parsed_date'])['distance'].sum().reset_index(name ='distance')
	#print 'Distance traveled in m:'
	#print userDistanceDF
	userDistanceDF.to_csv('out_distance.csv')
	userTimeDF = userDayTransportDF.groupby(['t_user_id','parsed_date'])['timediff'].sum().reset_index(name ='time')
	#print 'Time traveled in seconds:'
	#print userTimeDF
	userTimeDF.to_csv('out_time.csv')
	userSpeedDF = userDayTransportDF.groupby(['t_user_id','parsed_date'])['speed'].mean().reset_index(name ='speed')
	#print 'Average travel speed in m/s:'
	#print userSpeedDF
	userSpeedDF.to_csv('out_speed.csv')
	userAccelerationDF = userDayTransportDF.groupby(['t_user_id','parsed_date'])['acceleration'].mean().reset_index(name ='acceleration')
	#print 'Average travel acceleration in m/s2:'
	#print userAccelerationDF
	userAccelerationDF.to_csv('out_acc.csv')
	userBearingDF = userDayTransportDF.groupby(['t_user_id','parsed_date'])['bearing'].mean().reset_index(name ='bearing')
	#print 'Average compass bearing:'
	#print userBearingDF
	userBearingDF.to_csv('out_bearing.csv')

	#END of QUESTION1

def question2(userDayTransportDF):
	#The following part is inspired by Dr. Amilcar Soares code provided in Assignment-Pandas-ScikitLearn.pdf
	all_trajs = pandas.DataFrame()
	for myid in userDayTransportDF.t_user_id.unique():
		for mydate in userDayTransportDF.parsed_date.unique():
			for mymode in userDayTransportDF.transportation_mode.unique():
				myslice = userDayTransportDF[(userDayTransportDF['t_user_id'] == myid) & (userDayTransportDF['parsed_date'] == mydate) & (userDayTransportDF['transportation_mode'] == mymode)]
				# subtrajs with less than 10 points are dealt with in preprocessing
				if(myslice.shape[0]<1):
					continue
				#known issue: to_datetime will convert time to UTC thus there are days that may become split, but this also happens without UTC conversion
				#Suggested solution: take a time difference period threshold for splitting days instead of using 12AM
				myslice.parsed_time = pandas.to_datetime(myslice.parsed_time)
				newslice = myslice[['parsed_time','distance','speed','acceleration','bearing']].set_index('parsed_time')
				newslice = newslice.resample('D')
				means = newslice.mean()
				means.columns = ['distance_mean','speed_mean','acc_mean','bearing_mean']
				amin = newslice.min()
				amin.columns = ['distance_min', 'speed_min', 'acc_min', 'bearing_min']
				amax = newslice.max()
				amax.columns = ['distance_max', 'speed_max', 'acc_max', 'bearing_max']
				amedian = newslice.median()
				amedian.columns = ['distance_median', 'speed_median', 'acc_median', 'bearing_median']
				
				subtrajs = pandas.concat([means, amin], axis=1)
				subtrajs = pandas.concat([subtrajs, amax], axis=1)
				subtrajs = pandas.concat([subtrajs, amedian], axis=1)
				#overall std squared = (summation of multiple stds squared, which is the variance)/total number; eg. std^2 = (std(a)^2 + std(b)^2)/2 src:https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
				subtrajs['distance_std'] = sqrt((myslice['distance_std']**2).sum()/myslice.shape[0])
				subtrajs['speed_std'] = sqrt((myslice['speed_std']**2).sum()/myslice.shape[0])
				subtrajs['acc_std'] = sqrt((myslice['acc_std']**2).sum()/myslice.shape[0])
				subtrajs['bearing_std'] = sqrt((myslice['bearing_std']**2).sum()/myslice.shape[0])
				subtrajs['class'] = mymode
				all_trajs = all_trajs.append(subtrajs)
	all_trajs = all_trajs.dropna(axis=0, how='any')
	#print all_trajs
	print all_trajs.head()
	all_trajs.to_csv('out2.csv')
#print dataset.columns
#dataset = dataset.values

def plots(myDF):
	myDFcorr = myDF.corr()
	sns.heatmap(myDFcorr)
	plt.show()
	#convert to km so that it doesn't ruin the scale
	myDF["distance_mean"] = myDF['distance_mean']/1000 #km
	myDF["distance_std"] = myDF['distance_std']/1000 #km
	myDF["distance_median"] = myDF['distance_median']/1000 #km
	myDF["distance_max"] = myDF['distance_max']/1000 #km
	myDF["distance_min"] = myDF['distance_min']/1000 #km
	#parallel_coordinates(myDF, 'class')
	#plt.show()
	#plot the means using parallel coordinates
	meanDF = myDF.groupby(['class'], as_index=False).mean()
	print meanDF
	parallel_coordinates(meanDF, 'class')
	plt.show()
	print myDF
	#plot the std using parallel coordinates
	stdDF = myDF.groupby(['class']).std().reset_index('class')
	print stdDF
	parallel_coordinates(stdDF, 'class')
	plt.show()


def flatClassifier(myDF, myclassifier):
	skf = StratifiedKFold(n_splits=10)
	X = np.array(myDF.drop(['class'], axis=1))
	y = np.array(myDF['class'])
	skf.get_n_splits(X, y)
	acc = np.array([])
	for train_index, test_index in skf.split(X, y):
		myclassifier.fit(X[train_index], y[train_index])
		score = accuracy_score(myclassifier.predict(X[test_index]), y[test_index])
		acc = np.append(acc, score)
	return acc

def trainHC(df, className, myclassifier):
	X = np.array(df.drop([className], axis=1))
	y = np.array(df[className])
	#old code, doesn't use k-folds
	#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
	skf = StratifiedKFold(n_splits=10)
	skf.get_n_splits(X, y)

	classifiers = []
	test_class = None
	X_tests = None
	y_tests = None
	testclass = None
	counter = 0
	for train_index, test_index in skf.split(X, y):
		X_test = np.array(X[test_index])
		X_train = np.array(X[train_index])

		#remove last column, which is: "class"
		test_class_value = X_test[:,-1]
		X_test = X_test[:,:-1]
		X_train = X_train[:,:-1]
		myclassifier.fit(X_train, y[train_index])
		#this classifier looses indexes because it's being appended to numpy array so text is converted to numbers; this is a workaround for that
		try:
		    classifiers += list(myclassifier)
		except TypeError:
		    classifiers += list((myclassifier,))
		if(X_tests is None):
			test_class = test_class_value
			X_tests = X_test
			y_tests = y[test_index]
		else:
			#we need test_class to be able to compare the final results
			test_class =  np.append(test_class,test_class_value)
			X_tests = np.concatenate((X_tests, X_test), axis=0)
			y_tests = np.concatenate([y_tests, y[test_index]])
		counter+=1
	#counter is added as a workaround because randomforestclassifier is an array; thus affecting the length of the concatenated array
	return classifiers, X_tests, y_tests, test_class, counter

def hierarchyClassifier(myDF, trainSubDF, walkDF, finalDF, myclassifier):
	rfClassifiers1Trained, X_test , y_test , y_test_class , length = trainHC(myDF,'level1',myclassifier)
	rfClassifiers2Trained, X2_test, y2_test, y2_test_class, length2 = trainHC(trainSubDF,'level1_1',myclassifier)
	rfClassifiers3Trained, X3_test, y3_test, y3_test_class, length3 = trainHC(walkDF,'level2',myclassifier)
	rfClassifiers4Trained, X4_test, y4_test, y4_test_class, length4 = trainHC(finalDF,'finalclass',myclassifier)
	#np.set_printoptions(threshold=np.nan)

	#apply classifier on different levels
	accuracies =[]
	#loop on trained classifiers (10 folds => 10 trained classifiers)
	for n in range(length):
		predictions=[]
		for i in range(X_test.shape[0]):
			#n*10 is a workaround becuase numpy appends saves the classifier 10 times per iteration
			prediction = rfClassifiers1Trained[n].predict(X_test[i].reshape(1, -1))[0]
			
			if(prediction == 1):
				prediction = rfClassifiers2Trained[n].predict(X_test[i].reshape(1, -1))[0]
				
				if(prediction == 1):
					predictions.append('train')
				else:
					predictions.append('subway')
			else:
				prediction = rfClassifiers3Trained[n].predict(X_test[i].reshape(1, -1))[0]
				if(prediction == 1):
					predictions.append('walk')
				else:
					
					prediction = rfClassifiers4Trained[n].predict(X_test[i].reshape(1, -1))[0]
					predictions.append(prediction)
		#print predictions
		#print y_test_class
		#print predictions
		#print y_test_class
		accuracies.append(accuracy_score(predictions, y_test_class))
	return accuracies

if os.path.isfile('out2.csv'):
	print "Answers to questions 1 & 2 found, proceeding\n"
	DF = pandas.read_csv('out2.csv')
	del DF["parsed_time"]
	#uncomment the following line for the plots
	plots(myDF)
	#no need to binarize because I chose multinomial classifiers, but keeping it like this for proof of concept
	#prepare dataset for hierarchy
	myDF = DF.assign(level1=[1 if DF.iloc[i]["class"] in ["train","subway"] else 0 for i in range(len(DF.values))])
	trainSubDF = myDF[myDF["class"].isin(["train","subway"])]
	trainSubDF = trainSubDF.assign(level1_1=[1 if trainSubDF.iloc[i]["class"] == "train" else 0 for i in range(len(trainSubDF.values))])
	walkDF = myDF[~myDF["class"].isin(["train","subway"])]
	walkDF = walkDF.assign(level2=[1 if walkDF.iloc[i]["class"] == "walk" else 0 for i in range(len(walkDF.values))])
	finalDF = myDF[~myDF["class"].isin(["train","subway","walk"])]
	finalDF = finalDF.assign(finalclass=finalDF["class"])
	
	del trainSubDF['level1']
	del walkDF['level1']
	del finalDF['level1']
	HRFScore = hierarchyClassifier(myDF,trainSubDF, walkDF, finalDF,RandomForestClassifier())
	print 'Hierarchical Random Forest Score:'
	print sum(HRFScore)/float(len(HRFScore))
	FRFScore = flatClassifier(DF,RandomForestClassifier())
	print 'Flat Random Forest Score:'
	print FRFScore.mean()
	HDTcore = hierarchyClassifier(myDF, trainSubDF, walkDF, finalDF, DecisionTreeClassifier())
	print 'Hierarchical Decision Tree Score:'
	print sum(HDTcore)/float(len(HDTcore))
	FDTScore = flatClassifier(DF,DecisionTreeClassifier())
	print 'Flat Decision Tree Score:'
	print FDTScore.mean()
	###T-test ###
	statistic, HRFvsFRF = ttest_ind(HRFScore, FRFScore)
	print 'T-Test Score between Flat and Hierarchical Random Forests:'
	print HRFvsFRF
	statistic, HDTvsFDT = ttest_ind(HDTcore, FDTScore)
	print 'T-Test Score between Flat and Hierarchical Decision Trees:'
	print HDTvsFDT
	
elif os.path.isfile('preprocessed2.csv'):
	print "preprocessed dataset found, proceeding\n"
	userDayTransportDF = pandas.read_csv('preprocessed2.csv')
	question1(userDayTransportDF)
	question2(userDayTransportDF)
elif os.path.isfile('preprocessed.csv'):
	#file without features, just proper parsing
	print "preprocessed dataset found, proceeding\n"
	print "please wait: preprocessing dataset; part: 2\n"
	dataset = pandas.read_csv('preprocessed.csv')
	grouped1 = dataset.groupby(['t_user_id','parsed_date','transportation_mode'], as_index=False)
	userDayTransportDF = grouped1.apply(dayGroup)
	userDayTransportDF.to_csv('preprocessed2.csv')
	question1(userDayTransportDF)
	question2(userDayTransportDF)
else:
	print "please wait: preprocessing dataset; part: 1\n"
	dataset = pandas.read_csv('geolife_raw.csv')
###Parse DateTime & Replace it in Dataset As Date only, RUN ONCE ONLY, THIS IS PREPROCESSING###
	dataset = dataset[~dataset['transportation_mode'].isin(['motorcycle','run'])]
	dataset = dataset.assign(parsed_time=[parse(dataset.iloc[i]["collected_time"]) for i in range(len(dataset.values))])
	dataset = dataset.assign(parsed_date=[dataset.iloc[i]["parsed_time"].date() for i in range(len(dataset.values))])
	del dataset["collected_time"]
	dataset.to_csv('preprocessed.csv')
	#quick workaround because dates are already parsed, which will cause issues later
	dataset = pandas.read_csv('preprocessed.csv')
	print "please wait: preprocessing dataset; part: 2\n"
	grouped1 = dataset.groupby(['t_user_id','parsed_date','transportation_mode'], as_index=False)
	userDayTransportDF = grouped1.apply(dayGroup)
	userDayTransportDF.to_csv('preprocessed2.csv')
	question1(userDayTransportDF)
	question2(userDayTransportDF)