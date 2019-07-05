# 60daysofudacity
This repository contains the daily updates made during the 60daysofudacity challenge. This was an initiative from the Secure and Private AI Challenge Scholarship program.


# DAY 1 [1.6%] | 60
Begin with the #60daysofudacity challenge, completed the following:

* Lesson3: Introducing Differential Privacy [completed]
* Read the Book The Algorithmic Foundations of Differential Privacy, section 1: The Promise of Differential Privacy [page: 5 – 10].
* Working in the implementation of the project for the Lesson 3.

__What I learn:__
This was my first introduction to the field of Differential Privacy. As pointed out in the lectures, it is important to have a well defined framework, which allow us to define what is really to be private in the context of deep learning. Also, from the book, I can now see how the notion of privacy have been evolving. As with many field related in computer science, it is obvious to predict, that, in the future, the field will become more complex.

# DAY 2 [3.3%] | 60
 
* Implementation of the project 2 for the Lesson 3 completed.
* Added a plot for the project, where the data distribution can be compared using two databases.
* Taking a recap from Lesson 3.

__What I learn:__
This was a very interesting project, where we created a main database, containing a single feature with ones and zeros. Also, we implemented a function to create more databases from the main one, with one row missing per database (a total of 5000 dbs). I noticed how we select the probability distribution for the samples to be 50%. This give me the idea to plot the density distribution for the databases using different probabilities. In fact, I plot a standard DB with p = 0.5 against one with p = 0.7. This help me to understand how the probability parameter affected the creation of the databases.

![](plots/figure_2d.png)

# DAY 3 [5.0%] | 60
* Beginning with Lesson 4: Evaluating the Privacy of a Function [videos 1 and 2]
* Working in the initial project for this lesson.

__What I learn:__
It is interesting to see how some simple arithmetic functions, like sums, can be used to guest the identity of an individual in a database. This of course, makes necessary to address such issues. In the following day I will continue watching the lectures.

# DAY 4 [6.7%] | 60
* Project: Intro Evaluating The Privacy Of A Function completed
* Continuing with Lesson 4: Evaluating the Privacy of a Function [videos 3, 4, 5 and 6]

__What I learn:__
In this lesson I learn that it is possible to guest some of the distribution of the data applying simple arithmetic functions. Moreover, one can guest the individuals identity in a database. This means, we must implement the necessary mechanisms to guarantee the privacy in databases. _See notebook for day 4_

# DAY 5 [8.3%] | 60
* Evaluating The Privacy Of A Function using the iris data set.
* Working on projects [3, 4, 5 and 6] from Lesson 4: Evaluating the Privacy of a Function.

__What I learn:__
In this day, I implemented a function to calculate the sensitivity of the iris data set. Since we were working with a single feature, I evaluated the sensitivity of each feature using the sum query. I noticed how the sensitivity is affected for each feature when applying a simple sum operation. This is very interesting and shows how data can be susceptible when applying such operations. _See notebook for day 5_

# DAY 6 [10.0%] | 60

* Finish  Lesson 4: Evaluating the Privacy of a Function
* Completed projects all projects from this Lesson.
* Recap the Lesson.

__What I learn:__
In this lesson I learn about the implementation of a differencing attack over a database using the threshold query. Moreover, different functions can be applied in order to get information from databases. Also, data tend to be susceptible for such operations. _See notebook for day 6_

# DAY 7 [11.7%]
* Beginning with Lesson 5: Introducing Local and Global Differential Privacy [videos 1 – 4].
* Reading section 2:  [page: 11 – 15] from the Book The Algorithmic Foundations of Differential Privacy.

__What I learn:__
I learn about two types of privacy, which are local and global privacy. In the first method, the data is altered with some type of noise. This method guarantees more protection for the users, since the data itself is been altered. On the other hand, in global privacy, the noise is added to the output of the query, instead of the data itself as with local privacy. From this context I think, that in some scenarios, global privacy could be more effective, since local privacy has an inherent resource cost.
