# Insurance_Bill_Prediction
Insurance Bill Prediction

## Objective:
 To build a web application where demographic and health information of a patient is entered in a web form to predict charges.
 
## Tools Used:

 ### 1. PyCaret: 
PyCaret is an open source, low-code machine learning library in Python to train and deploy machine learning pipelines and models in production. PyCaret can be installed easily using pip.
 
 ### 2. Flask:
Flask is a framework that allows you to build web applications. A web application can be a commercial website, a blog, e-commerce system, or an application that generates predictions from data provided in real-time using trained models. If you don’t have Flask installed, you can use pip to install it.
 
 ### 3. GitHub
GitHub is a cloud-based service that is used to host, manage and control code. Imagine you are working in a large team where multiple people (sometime hundreds of them) are making changes. PyCaret is itself an example of an open-source project where hundreds of community developers are continuously contributing to source code. If you haven’t used GitHub 
before, you can sign up for a free account.

 ### 4. Heroku
Heroku is a platform as a service (PaaS) that enables the deployment of web apps based on a managed container system, with integrated data services and a powerful ecosystem. In simple words, this will allow you to take the application from your local machine to the cloud so that anybody can access it using a Web URL. In this tutorial we have chosen Heroku for deployment as it provides free resource hours when you sign up for new account.

![c1](https://user-images.githubusercontent.com/43957442/133999985-758fcc3b-6610-44fc-940a-3f699d68c2b9.PNG)

## Tasks:
 
1. Train and validate models and develop a machine learning pipeline for deployment.
2. Build a basic HTML front-end with an input form for independent variables (age, sex, bmi, children, smoker, region).
3. Build a back-end of the web application using a Flask Framework.
4. Deploy the web app on Heroku. Once deployed, it will become publicly available and can be accessed via Web URL.

👉 Task 1 — Model Training and Validation
 
Training and model validation are performed in Integrated Development Environment (IDE) or Notebook either on your local machine or on cloud. In this tutorial we will use PyCaret in Jupyter Notebook to develop machine learning pipeline and train regression models. 

The first experiment is performed with default preprocessing settings in PyCaret (missing value imputation, categorical encoding etc). The second experiment has some additional preprocessing tasks such as scaling and normalization, automatic feature engineering and binning continuous data into intervals. See the setup example for the second experiment:

Experiment No. 2from pycaret.regression import *r2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])
 

The magic happens with only a few lines of code. Notice that in Experiment 2 the transformed dataset has 62 features for training derived from only 7 features in the original dataset. All of the new features are the result of transformations and automatic feature engineering in PyCaret
 

### Sample code for model training and validation in PyCaret:

#### Model Training and Validation 
  lr = create_model('lr')
 

Notice the impact of transformations and automatic feature engineering. The R2 has increased by 10% with very little effort. We can compare the residual plot of linear regression model for both experiments and observe the impact of transformations and feature engineering on the heteroskedasticity of model.

#### plot residuals of trained model
  plot_model(lr, plot = 'residuals')

![pycaret-web-app-6](https://user-images.githubusercontent.com/43957442/134000957-2609247f-c378-40a6-931e-9098e79d3aa0.png)


 

Machine learning is an iterative process. Number of iterations and techniques used within are dependent on how critical the task is and what the impact will be if predictions are wrong. The severity and impact of a machine learning model to predict a patient outcome in real-time in the ICU of a hospital is far more than a model built to predict customer churn.


#### save transformation pipeline and model 
  save_model(lr, model_name = 'c:/username/ins/deployment_28042020')


When you save a model in PyCaret, the entire transformation pipeline based on the configuration defined in the setup() function is created . All inter-dependencies are orchestrated automatically.

👉 Task 2 — Building Web Application
 
Now that our machine learning pipeline and model are ready we will start building a web application that can connect to them and generate predictions on new data in real-time. There are two parts of this application:

Front-end (designed using HTML)
Back-end (developed using Flask in Python)
 

### Front-end of Web Application
 
Generally, the front-end of web applications are built using HTML which is not the focus of this article. We have used a simple HTML template and a CSS style sheet to design an input form. 

CSS Style Sheet

CSS (also known as Cascading Style Sheets) describes how HTML elements are displayed on a screen. It is an efficient way of controlling the layout of your application. Style sheets contain information such as background color, font size and color, margins etc. They are saved externally as a .css file and is linked to HTML but including 1 line of code.
 
### Back-end of Web Application
 
The back-end of a web application is developed using a Flask framework. For beginner’s it is intuitive to consider Flask as a library that you can import just like any other library in Python. See the sample code snippet of our back-end written using a Flask framework in Python.
 

If you remember from the Step 1 above we have finalized linear regression model that was trained on 62 features that were automatically engineered by PyCaret. However, the front-end of our web application has an input form that collects only the six features i.e. age, sex, bmi, children, smoker, region.
How do we transform 6 features of a new data point in real-time into 62 features on which model was trained? With a sequence of transformations applied during model training, coding becomes increasingly complex and time-taking task.
In PyCaret all transformations such as categorical encoding, scaling, missing value imputation, feature engineering and even feature selection are automatically executed in real-time before generating predictions.
Imagine the amount of code you would have had to write to apply all the transformations in strict sequence before you could even use your model for predictions. In practice, when you think of machine learning, you should think about the entire ML pipeline and not just the model.

### Testing App

One final step before we publish the application on Heroku is to test the web app locally. Open Anaconda Prompt and navigate to folder where ‘app.py’ is saved on your computer. Run the python file with below code:

  #python app.py
 

Once executed, copy the URL into a browser and it should open a web application hosted on your local machine (127.0.0.1). Try entering test values to see if the predict function is working. In the example below, the expected bill for a 19 year old female smoker with no children in the southwest is $20,900.

![pycaret-web-app-12](https://user-images.githubusercontent.com/43957442/134001431-cf167cbb-2098-4bca-a1a9-a9f1c3cfb358.png)


👉 Task 3 — Deploy the Web App on Heroku
 
Now that the model is trained, the machine learning pipeline is ready, and the application is tested on our local machine, we are ready to start our deployment on Heroku. There are couple of ways to upload your application source code onto Heroku. The simplest way is to link a GitHub repository to your Heroku account. 

### requirements.txt
requirements.txt  file is a text file containing the names of the python packages required to execute the application. If these packages are not installed in the environment application is running, it will fail.

### Procfile
Procfile is simply one line of code that provides startup instructions to web server that indicate which file should be executed first when somebody logs into the application. In this example the name of our application file is ‘app.py’ and the name of the application is also ‘app’. (hence app:app)

Once all the files are uploaded onto the GitHub repository, we are now ready to start deployment on Heroku. Follow the steps below:

Step 1 — Sign up on heroku.com and click on ‘Create new app’
Step 2 — Enter App name and region
Step 3 — Connect to your GitHub repository where code is hosted
Step 4 — Deploy branch


## FLOW CHART:
![Untitled Diagram drawio](https://user-images.githubusercontent.com/43957442/134016410-014367d4-8dc4-4617-8d13-68d76d4c08b2.png)



### App is published to URL: https://pycaret-insurance.herokuapp.com/
 
