# ./svm-predict ../Dataset/mnist_libsvm/test_scale.csv train_scale.csv.model output | grep -o "[0-9.]*%"
# ./svm-train -s 0 -t 2 -c 1 -g 0.05 ../Dataset/mnist_libsvm/train_scale.csv gaussian
# ./svm-train -s 0 -t 2 -c 1 -g 0.05 -v 10 ../Dataset/mnist_libsvm/train_scale.csv gaussian_cross_1
cd libsvm-3.22
make
cd ../
cross_valid_file="cross_valid_accu.txt"
train_accu_file="train_accu.txt"
test_accu_file="test_accu.txt"

rm $cross_valid_file
touch $cross_valid_file

rm $train_accu_file
touch $train_accu_file

rm $test_accu_file
touch $test_accu_file

# for C in 10
for C in 0.00001 0.001 1 5 10
do
	echo $C
	./libsvm-3.22/svm-train -s 0 -t 2 -c $C -g 0.05 -v 10 Dataset/mnist_libsvm/train_scale.csv | grep -o "[0-9.]*%"  >> $cross_valid_file
	./libsvm-3.22/svm-train -s 0 -t 2 -c $C -g 0.05 Dataset/mnist_libsvm/train_scale.csv model_SVM/model_gaussian_$C | grep -o "[0-9.]*%"  >> $train_accu_file
	./libsvm-3.22/svm-predict Dataset/mnist_libsvm/test_scale.csv model_SVM/model_gaussian_$C pred_SVM/pred_$C.csv  | grep -o "[0-9.]*%"  >> $test_accu_file
done

