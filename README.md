# deffect-predictions


Predictive models for software bugs in Java source code.

This repository supplements the blog found [here](https://kevin-morales.github.io/deffect-predictions/).
It contains .Rmd, and .R files which have the following code:

* Code for analysis and plots.
* Contains the code that trains 4 different models.
* Feature importance for the best model.
* Data frames containing the accuracy and the area under the curve for each model.

## Running the Code

* Clone the repository.

``~~~
git clone https://github.com/Kevin-Morales/deffect-predictions.git
~~~

* Navigate to the directory where it was cloned.
* Download the [Unified Bug Dataset](http://www.inf.u-szeged.hu/~ferenc/papers/UnifiedBugDataSet/)
* Copy the file "Unified-class.csv" to the directory of the cloned repository.
* Point R or R Studio to the directory.
* Open the file named "analysis-and-interrogation.R," highlight all of the code, and press control + r to run it.

