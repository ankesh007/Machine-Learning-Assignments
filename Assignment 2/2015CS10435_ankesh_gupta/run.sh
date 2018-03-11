#!/bin/bash

prefix="Clean_"
prefix2="Clean_2"
if [ $1 -eq 1 ] 
	then		
    # echo This is question 1.
	if [ $2 -eq 1 ]
		then
		python3 CleaningData.py $3 $prefix$3
		python3 NaiveBayes.py 1 $prefix$3 model_NaiveBayes/model_1 $4		
		rm $prefix$3

	elif [ $2 -eq 2 ]
		then
		python3 Stemmer.py $3 $prefix$3
		python3 NaiveBayes.py 1 $prefix$3 model_NaiveBayes/model_2 $4		
		rm $prefix$3

	elif [ $2 -eq 3 ]
		then
		python3 Stemmer.py $3 $prefix$3
		python3 NaiveBayesBetter.py 1 $prefix$3 model_NaiveBayes/model_3 $4		
		rm $prefix$3

	else
		echo Unknown option.
	fi

elif [ $1 -eq 2 ]
	then
	# echo This is question 2.
	if [ $2 -eq 1 ]
		then
		python3 SVM.py 1 $3 model_SVM/SVM_my_model $4

	elif [ $2 -eq 2 ]
		then
		python3 TestDatasetForLibSVM.py $3 $prefix$3
		./libsvm-3.22/svm-scale -l 0 -u 1 $prefix$3 > $prefix2$3
		./libsvm-3.22/svm-predict $prefix2$3  model_SVM/model_linear_1 $4 
		rm $prefix$3
		rm $prefix2$3 

	elif [ $2 -eq 3 ]
		then
		python3 TestDatasetForLibSVM.py $3 $prefix$3
		./libsvm-3.22/svm-scale -l 0 -u 1 $prefix$3 > $prefix2$3
		./libsvm-3.22/svm-predict $prefix2$3  model_SVM/model_gaussian_10 $4  
		rm $prefix$3
		rm $prefix2$3 

	else
		echo Unknown option.
	fi
else 
     	echo Unknown option.
fi