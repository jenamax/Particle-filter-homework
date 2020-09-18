## Particle Filter
An alternative nonparametric technique is known as particle filter. Particle filters represent posteriors by a random sample of states, drawn from the posterior. Such samples are called particles. 

----

Specific strategies exist to reduce the error in particle filters. Among the most popular ones are techniques for reducing the variance of the estimate that arises from the randomness of the algorithm, and techniques for adapting the number of particles in accordance with the complexity of the posterior.


----
In this repo we introduced a solution to a localization problem and implemeted some methods. To start the code you need to run the script main.py in scripts folder, don't forget to install all dependencies before.

##### Whole pipeline: 
- Motion model
- Sensor Model
- Resampling
 and some files for testing that methods(sensor_test.py, motion_model_test.py)
