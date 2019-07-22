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

# DAY 8 [13.3%] | 60
* Continuing with Lesson 5 [videos 5 – 7].
* Working on the Project: Implement Local Differential Privacy

__What I learn:__
Today I learn about two types of noise, which can be added in global privacy: Gaussian and Laplacian. From this type of noises, at the time, the Laplacian noise is more widely used, due to its relative easy calculation. Also, the formal definition of privacy implement two important factors: epsilon and delta. The former measures the difference in distributions from the original data and the data with missing entries. Meanwhile, delta represent the probability of leaking extra information. For that reason, the usual values to delta are very tiny or zero.

# DAY 9 [15.0%] | 60
* Project: Implement Local Differential Privacy [completed]
* Applying variations to the project [1] [adding different values of epsilon, plots, run more tests]

__What I learn:__
Today I learn about how the amount of data can impact the queries over a data base. More precisely, I set up an experiment, where the mean query was executed with different entries in a database. I notice that, each time the entries increase, the approximation for the real value of the mean query was more close. Meaning the differential data come more close to the real result of the query on the real data. When repeating this experiment multiples times I observed the same results. At first, with less entries, the distance in the results where big. However, the more entries, the more close the results were to the real ones. This result reaffirms the discussion on the lectures. _See notebook for day 9_
![](plots/figure_9d.png)

# DAY 10 [16.7%] | 60
* Adding final variations to the Project from Lesson 5 [plot more functions]
* Beginning to work on the final project for Lesson 5.

__What I learn:__
Today I decided to continue with the experiments from the last project. This time, I added four extra queries: cumulative sum, random sum, logarithm sum and standard deviation. After running the experiments, I noticed that using the cumulative query, one can approximate the real query on the data base with little entries. However, increasing the entries, will also increase the gap between the queries. This is also true for the random sum and logarithm sum queries. On the contrary, the standard deviation query, acts in the same fashion as the mean query. Where with more data, the results will better approximate. This help me to understand that, not all queries behave in the same ways. Therefore, when applying global privacy, one must careful consider the used mechanisms. _See notebook for day 10_
![](plots/figure_10d.png)

# DAY 11 [18.3%] | 60
* Finished final project for Lesson 5: Create a Differentially Private Query.
* Recap from Lesson 5.

__What I learn:__
Global and local privacy are two types of privacy mechanism which can be implemented. I think that, in the case of deep learning, one could be more inclined to use global privacy, since it only affects the outputs of the model. In contrast, with local privacy, one must change the data. This process could be expensive in some settings. For example, with many images. However I think that local privacy can be applied in the context of machine learning, when cost of transforming the data is low. _See notebook for day 11_

# DAY 12 [20.0%] | 60
* Entering a Kaggle competition: Generative Dog Images. | Goal: make a submission and apply the learned on the DLND program. 
* Creating the data set from the training data using torch vision.
* Working in some potentially architectures.

__What I learn:__

Today I decided to take part in a Kagle competition. This competition is about creating a GAN model to generated dog images. To start, this data set is composed by a total of 20579 images. However, not all the images displays the targets (dogs). There are some samples where other labels are presents, like: persons, etc. Also, it is very interesting to see how GAN’s can be applied to different problems.

# DAY 13 [21.7%] | 60
* Having an interesting discussion about research topics in machine learning.
* Training a baseline GAN model to generate dog images.
* Sending my initial submission for the Kaggle competition: Generative Dog Images.
* Planning on future improvements for the baseline model.

__What I learn:__

Today, I learned how convolutional neural networks can be applied to recognize sign language characters. It is interesting to see how convolutional networks models can achieve great accuracy in such tasks. Regarding the training of my GAN model, I noticed how the features (filters) can play an important role at the moment to generate quality images. In fact, applying variations can lead to more natural results. However, other aspects like the number of convolutional layers can also affect the learned representations. Therefore I think a gradual approach should be considered, where in each stage, a set of layers / features are added, until get a desired result according with the available  computational resources.

![](plots/figure_d13.png)

# DAY 14 [23.3%] | 60* #60daysofudacity

* Beginning with Lesson 6: Differential Privacy for Deep Learning.
* Studying lectures: 1, 2 and 3.
* Reading the paper: Generating Diverse High-Fidelity Images with VQ-VAE-2: https://arxiv.org/pdf/1906.00446.pdf

__What I learn:__

Today, I learned about how to generate labels in a differentially private setting. More precisely, this technique allow us to generate labels using external classifiers. Of course, this classifiers must belong to the same category we want to obtain the labels. In this case, we have our data which we do not have the labels, and we will use these classifiers to generate our labels. However, in order to assure the privacy component, we will add some degree of epsilon (privacy leak) over the generation of the labels. This will be used as part of a Laplacian noise mechanism (we can use Gaussian too). In this way, we are obtaining the labels for our local data set without compromising the privacy of the individuals in the external data sets. Also, from the paper I read today, I learned about the VQ-VAE-2 model. This generative model is able to generate realistic images using a vector quantized mechanism.

# DAY 15 [25.0%] | 60
* Continuing with Lesson 6.
* Studying lectures: 4, 5.
* Reading the suggested paper material: Deep Learning with Differential Privacy [completed].

__What I learn:__

Today, I learned about PATE analysis, a technique which allow us to analyze how much information is leaked out. In this context, PATE analysis will measure the degree of the epsilon parameter. However, it is also possible to apply differential privacy to the models instead. In particular, a variation of the SGD algorithm can be used. This DPSGD calculate the gradients from a random subset of the data. Then it clips the gradients using the L2 norm. Next, it averages the gradients and add noise. Here, the noise is one of the mechanism to assure privacy. Finally it moves the gradients in the opposite direction of the average gradients (which have some degree of noise). As we can see, this algorithm can be used to train a model at the same time that maintains the user privacy.

# DAY 16 [26.7%] | 60
* Adding improvements to the baseline GAN model for the Kaggle competition: Generative Dog Images.
* Training the new baseline model.
* Sending submission.
* Obtaining better MiFID score: from _128.21376_ to __114.34636.__

__What I learn:__

Today, I put into practice the techniques I learned in the DLND program about GANS. This allowed me to improve my initial baseline model, which is developed in Pytorch. The main idea was varying the complexity of the kernels (filters) in the model. Since the kernels will increase the model’s complexity, thus allowing the model to learn better representations. Also, in order to gain better stability during training, I applied diverse regularization techniques. Finally, I learned about a variation for the FID score called MiFID. This metric considers the model’s memorization. This will allow to evaluate the model not only to generate images, but also the diversity of the generated images.

![](plots/figure_d16.png)

# DAY 17 [28.3%] | 60
* Continuing with Lesson 6.
* Studying lectures 7 and 8 .
* Working on the final project for Lesson 6, using a data set.
* Watching the webinar from OpenMined: https://www.youtube.com/watch?v=9D_jxOMZmRI

__What I learn:__

Today I learned more aspects about differential privacy. In particular, DP, interacts with other technologies such as encryption. Also DP allow us to learn useful features without compromising the user privacy. In this context, the different techniques (algorithms) usually adds some kind of noise to the output model or the data itself. Also, open source projects, have a tremendous impact in the development of different technologies, as seen in the webinar. Contribution is an important factor as well, since there are many opportunities in which one can contribute to projects. Finally, feedback is an important component inside the development of open source software. 

# DAY 18 [30.0%] | 60
* Recap from Lesson 6.
* Continuing working on the final project for Lesson 6: Defining seven phases, which covers all the content from the Lesson.

__What I learn:__

Today I take a recap from Lesson 6. I learned about a mechanism to generate labels using noise. In this case, two types of noise were described: Laplacian and Gaussian. Also, one can combine this technique with other classifiers. For example, if we have an unlabeled data set, we can use external data to generate labels. However, in order to maintain privacy, we ask the data owners to generate the labels from our data. Of course, the data must come from the same domain. In this way, we can generate labels without compromising the data privacy. We can also evaluate the generated labels in terms of the degree of epsilon (privacy leak) using PATE analysis. Finally, I am working in a project which involve all the material from the Lesson. Concretely, I will generate labels for a data set using the learned techniques.

# DAY 19 [31.7%] | 60
* Continuing working on the final project for Lesson 6: Phase One: The data set.
* Improving remote and local data simulation.
* Adding plots for the data.

__What I learn:__

Today I implemented the concepts of remote and local data sets discussed in Lessons 6. In this setting, we have a local data set for which we do not have labels. Therefore, we would like to use a set of remote data sets in order to train a local model. However, we cannot have access to these data sets directly. For example, the data sets can contain sensible data from patient records. Thus, it is important to define a safe procedure to access them. This will be addressed in the next phases. Now, back to the remote and local data sets, I selected the digits data sets as our main data. Then, I randomly divided the data into eleven blocks. The first ten will correspond to the remote data sets, meanwhile, the last one will be our local data. Since we will need a structure, I used dictionaries. Finally, I built a plot function to display the data set.

![](plots/figure_19ad.png)

![](plots/figure_19bd.png)

# DAY 20 [33.3%] | 60
* Continuing working on the final project for Lesson 6: Phase Two: Defining external classifiers.
* Selecting a set of ten classifiers to train on the remote data sets.
* Training external classifiers.

# What I learn:

Continuing with the concepts from Lesson 6: “...since we cannot access the remote data sets directly, we can instead use trained classifiers from those data sets to label our local data set.”. In order to get more diversity, different classifiers were used. Also, since we previously divided the data into 11 blocks, we have little data. Therefore techniques such as: cross validation will not be used. Instead, we will use the classifiers with their define set of hyper parameters. Then the training process began. In general, I observed that some models easily get a high accuracy, meanwhile others get a low one. However, we cannot directly conclude about what model are the best here, due to the lack of hyper parameter tuning. Also, we would like to keep in mind the famous No Free Lunch Theorem.

# DAY 21 [35.0%] | 60
* Continuing working on the final project for Lesson 6: Phase Three: Generate predictions to our local data set.
* Generating predictions to our local data set using the classifiers.
* Applying Laplacian noise with epsilon = 0.1 to the new generated labels.

__What I learn:__

Continuing with the implementation of the concepts from Lesson 6: “… we may use trained classifiers on remote data sets to labels ours. However, even with this mechanism, there are still ways in which we can guess the real values from the external data sets using the classifiers parameter’s”. Indeed, as we have seen in past lessons, it is totally possible to use some queries over the data sets to break privacy. In this case, the same can be applied to the classifiers (algorithms). In particular, if we use neural networks, we can use the raw gradients to obtain such information. Hopefully, we can add a noise degree over the label generation process. To be more precisely, this noise will represent the value of privacy we want to keep. As seen in the literature, this value correspond to the epsilon parameter. Now, regarding the noise, we can use different functions to generate it. However, as discussed in class, the more efficient, in terms of computationally cost and implementation is the Laplacian noise. Therefore we will apply the Laplacian noise, also, we set the value to 0.1.
![](plots/figure_21d.png)

# DAY 22 [36.7%] | 60

* Continuing working on the final project for Lesson 6: Phase Four: Defining a local model.
* Defining a pytorch model to be used locally.
* Training the local model on the generated data.

__What I learn:__

Continuing with the implementation of the concepts from Lesson 6: “… and then, after we have our local labels generated with differential privacy, now, we can train our local model, without compromising the the remote data sets.”. Therefore, today, I defined our local model in Pytorch. For this data set, I implemented a shallow network. Then, I proceed with the training process. However, instead of using the real labels from the data set, I use the generated labels. It is interesting to note, how these labels have been generated. In a sense, the are directly dependent on the external classifiers and the data sets. However, the differential mechanism applied guarantees that we can not break the privacy of the remote data sets. Furthermore, we have now an extra parameter which controls the degree of privacy. Of course, one can argue that, if the same person is carrying out the analysis, this person would also have access to the epsilon value. This discussion also arises in the privacy book: “The Algorithmic Foundations of Differential Privacy”. Hopeful, as one can guest, there are different forms in which we can assure the anonymity of the epsilon value. Therefore we can still guarantee privacy.

![](plots/figure_22d.png)

# DAY 23 [38.3%] | 60
* Meeting with the sg_latin group.
* Reading the paper: Improved Techniques for Training GANs: https://arxiv.org/pdf/1606.03498.pdf

__What I learned:__

Today,  we discussed about the current projects we have in the sg_latin group.  Also, we proposed new improvements to apply over the current project.  This project have the aim to apply differential privacy techniques.  Also, I learn about techniques that can be used to improve the  performance of a GAN model, thus, allowing the model to converge much  faster. This techniques are: feature matching, mini batch  discrimination, historical averaging, one sided label smoothing and  virtual batch normalization. Each technique address a particular element  from the training process. Finally, applying these techniques have a  positive impact over the quality of the generated images. Thanks to our classmates from the sg_latin group for made the meeting possible today.

# DAY 24 [40.0%] | 60*
* Reading the paper: A Unified Approach to Interpreting Model Predictions: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
* Implement a pytorch model, alongside the training function for the sg_latin project.

__What I learn:__

Today, I learned about an interesting research topic in deep learning: interpretability. This term is refereed to the model explaining capacity. For example, in a medical setting, alongside accuracy, it is also desirable to know the underlying mechanism that lead the classification criteria of the model. Also,  there are a relationship between complexity and explanation. For example, very complex models like deep networks are more difficult to explain, due to the many parameters they have. In contrast, more simple models, are easier to explain. Therefore there is a trade-off between complexity and explanation. This trade-off must be taken into consideration when dealing with applications that require additional explanations from the model. Also, I contribute to the sg_latin project, which main goal is to apply differential privacy. Concretely, my contributions were focused on implement a pytorch model alongside the training and evaluation process, using the provided code: https://github.com/rick1612/House-Price.
