# Implementing Regression Algorithms and GDA

The assignment is level 1 introduction to ***regression*** algorithms and ***Gaussian Discriminant Analysis(GDA)*** . More can be read on *ProblemStatement.pdf*.

The assignment consists on 4 parts. The 4 parts are placed in folder *Question {1,2,3,4}* respectively. ***Report.pdf*** captures various nuances of assignment and nice observations and inferences. 

A generalized ***Regression*** class was created, which by default supported *Linear Regression*. Other necessary *Regression* classes can inherit it as base class and ***override*** necessary methods. Look at code for better understanding.


*To run the code, type in linux shell:*

Question 1:
``` 
python3 <script> <path_x> <path_y> <draw_surface> <draw_contour> <draw_hypothesis> <solve_analytically> <learning_rate>
```

Question 2:
``` 
python3 <script> <x_data_path> <y_data_path> <tau>
```

Question 3:
``` 
python3 <script> <x_data_path> <y_data_path> <epsilon-optional>
```

Question 4:
``` 
python3 <script> <x_data_path> <y_data_path> <Draw_Other_quadratic_part>
```

*Note*: 
1. Install necessary packages whenever necessary.
2. You can turn off the plots as required by giving the requisite flags in Question 1.