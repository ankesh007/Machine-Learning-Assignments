# Implementing Naive Bayes and Support Vector Machines(SVM)

The assignment is level 1 introduction to ***Naive Bayes Classifier*** algorithms and ***SVM*** . More can be read on *ProblemStatement.pdf*.

The assignment consists on 2 parts. The trained models for the same can be found at:[Model](https://drive.google.com/open?id=1p4G8OKkiDIdbl0lVhqRG1OgxNayQ2HDE) 


***Multinomial Naive Bayes*** was implemented for text classification and linear kernel SVM was implemented. Also, experiments were run on [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm) *linear* and *gaussian* kernel.

For optimising self-implemented SVM, [this](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf) paper was referred.

*To run the code for prediction, type in linux shell:*

``` 
./run.sh <Question_number> <model_number> <input_file_name> <output_file_name>
```

To run individual files:
```
python3 <script.py> [many flags and names]
```


*Note*: 
1. `many flags and names` => verbose description is printed when running the script without them.
2. `pip3` install necessary packages whenever required to run.
3. The code has been tested on *Ubuntu 16.04* distribution of Linux.
4. `Plots/mnist_misclassified` has interesting misclassified mnist digits.
5. Better scripts are in `2015CS10435_ankesh_gupta`

