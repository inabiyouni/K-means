# K-means
a k-means algorithm which can use different distance functions in calculating SSN

Three data set are provided for evaluation as follow with the following names:
Human Activity Recognition: X_train.txt, y_train.txt, X_test.txt, y_test.txt
Iris: X_iris.txt y_iris.txt
Banknote: X_banknote.txt, y_banknote.txt

To run the program you can type:
python Q3_a argumnets

results which are the estimated clasess will be written on a txt file after each run of the program

Arguments for the command should be provided after these keys (some of them are optional:

-itr1  : name of a file containing attribute values
-itr2   : name of a file containing class values

OPTIONALS:
-its1  : name of a file containing attribute values
-its2   : name of a file containing class values
-f    : distance function name as "eucli" or "city" or "cosine" or "fun1" or "fun2" (DEFUALT: "eucli")
-K	: number of desired cluster (DEFAULT: number of 	clusters in the data set)
-r	:number of repetitions or restarting the program(DEFAULT: 1) 
-a	:visualization mode which can be "false" or "true" (DEFAULT: "false")
-ri	:random initialization of the custer centers which can be "false" or "true" (DEFAULT: "false") NOTE: in true mode the program will pick the centers furthest possible from each other.

sample running commands:

python K-means.py -itr1 X_iris.txt -itr2 y_iris.txt -r 5 -a false -ri true -f fun2

python K-means.py -itr1 X_train.txt -itr2 y_train.txt -its1 X_test.txt -its2 y_test.txt -r 1 -a false -ri false -f fun1


