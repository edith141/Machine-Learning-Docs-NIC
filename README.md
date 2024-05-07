
# Python - ML & Analytics


### Machine Learning Basic Terminology-

Before seeing the basic temrinologies used in Machine Learning, let us first see the basic steps / life-cycle of building a machine learning model- 

1.  **Collecting data**: Be it the images, text files, numberic, etc., this step is mostly the first step towards building a model. The better the variety, quality,  and volume of relevant data, better the learning prospects for the machine becomes.
2.  **Preparing the data**: Any machine learning/analytical process thrives on the quality of the data used. We need to preprocess the data before using it for learning and this may inlcude cleaning, filtering, augmentation, etc. (More on this later).
3.  **Training a model**: This step involves choosing the appropriate algorithm and representation of data and choice of model for the data and use case. The cleaned data is usually split into two parts – train and test; the training data, as the name suggests, is used for training the model whereas the test data, is used as a reference / testing.
4.  **Evaluating the model**: To test the accuracy, the second part of the data (holdout / test data) is used. A better test to check accuracy of model is to see its performance on data which was not used at all during model build, hence we keep some data completely isolated exclusively for this purpose. 

### Basic Terminology and Machine Learning Types
#### Supervised Learning / Predictive 
In supervised learning, the algorithm learns from labeled data to make predictions about unseen or future data points. These predictions are based on the patterns and relationships learned from the training data. Therefore, supervised learning algorithms are used to build predictive models that can accurately predict the target variable for new input data. 

Examples of supervised learning include:

1.  **Email Spam Detection**: Classifying emails as spam or non-spam based on features such as sender, subject, and content.
    
2.  **Medical Diagnosis**: Predicting the presence or absence of a disease based on patient symptoms and medical test results.
    
3.  **Stock Price Prediction**: Forecasting future stock prices based on historical price data, market trends, and other relevant factors.
    
Supervised learning is widely used across various domains for tasks such as classification, regression, and pattern recognition.

### Unsupervised Learning / Descriptive

In unsupervised learning, the algorithm learns from unlabeled data, aiming to discover hidden patterns or structures within the data. Unlike supervised learning, there are no explicit output labels provided during training, and the algorithm must infer the underlying structure based solely on the input data.

Examples of unsupervised learning include:

1. **Clustering**: Grouping similar data points together based on their inherent characteristics. For example, clustering customers based on purchasing behavior to identify market segments.

2. **Dimensionality Reduction**: Reducing the number of input variables while preserving essential information. Principal Component Analysis (PCA) is a common technique used for dimensionality reduction in unsupervised learning.

3. **Anomaly Detection**: Identifying unusual patterns or outliers in data that deviate from normal behavior. This can be applied in fraud detection, network security, or fault detection in industrial processes.

Unsupervised learning is valuable for exploratory data analysis, data preprocessing, and uncovering hidden patterns in large datasets where labeled data may be scarce or unavailable.

## Linear Regression

Linear regression is a supervised machine learning algorithm used to predict continuous values, such as sales figures or housing prices. It analyzes the relationship between two or more variables, where one variable, known as the dependent variable, is predicted based on the values of other variables, called independent variables.

The fundamental concept behind linear regression is to model the linear relationship between the independent and dependent variables. This relationship is represented by a straight line, known as the **regression line**, which shows how the value of the dependent variable changes as the independent variable(s) vary. 

![final sklearn](https://github.com/edith141/LinearReg-scratch/raw/main/final%20result_sklearn.png)
For example, in the above image, the red values are the values predicted by the Linear Regression model and the green values are the original data values (input). The red line is the regression line that can be used to predict future values.

The equation of the regression line is typically expressed as:
 
**`y = wx + b`**

Where:
- **X**: The independent variable.
- **Y**: The dependent variable that the model will predict.
- **Weight (w)**: The coefficient for the independent variable X. In machine learning terminology, these coefficients are referred to as weights.
- **Bias (b)**: The Y-intercept, which offsets all predicted values of Y. In machine learning, this is known as the bias term.

Linear regression is a versatile tool used in various fields, including economics, finance, and healthcare, for modelling relationships and making predictions based on observed data. 

**For example,
Predictive Analytics in Marketing**: Linear regression is widely used in marketing to predict customer behaviour and optimise marketing strategies. For example, businesses use linear regression to analyze historical sales data and forecast future sales trends, allowing them to adjust marketing campaigns and allocate resources effectively.

### Libray we can use for Linear Regression (in Python): 
**Library Name**: Scikit-learn

**Link**: [Scikit-learn](https://scikit-learn.org/)

**Module to use**: `sklearn.linear_model.LinearRegression`

Among many other modules for machine learning, Scikit-learn provides the `LinearRegression` module, which is used to implement linear regression models. We can start with linear regression by importing this module from scikit-learn and using it to fit a linear regression model to your data.

**Dataset Example**: We can use the Boston Housing dataset, which is already included in scikit-learn and provides information about housing prices in Boston. 

**Dataset Name**: California Housing Dataset

We can load this dataset directly from scikit-learn using the following module: 
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```

<br></br>

*Example implementation from scratch and more explanation -* https://github.com/edith141/LinearReg-scratch



## Logistic Regression

Logistic regression is a supervised machine learning algorithm used for binary classification tasks, such as spam detection or medical diagnosis. It models the probability that a given input belongs to a particular class by fitting a logistic function to the input features.

The logistic regression model calculates the probability of the input belonging to a certain class using the logistic function, which maps any input value to a value between 0 and 1. This probability is then used to make predictions, with a threshold applied to determine the predicted class label.

For example, we can use a  **logistic  (sigmoid) function**. This also uses an equation as its representation. The input value is combined with the coefficients (wts and bias) to predict an o/p- y.

Now, in terms of input "x" and weight "w", we can write this as:
$$ y =g(w.x) = \frac{1}{1 +e^(-w.x^)} $$

Where,
- **X**- The independent variable.

- **Y**- The dependent variable (the model will predict Y values).

- **Weight (w)**-  The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as  _weights_.

Logistic regression is widely used in various domains for binary classification tasks where the outcome variable has two possible outcomes.

**For example-**
**Credit Risk Analysis**: Logistic regression is used in finance to assess the credit risk associated with loan applicants. By analyzing various features of the applicant, such as credit score, income, and debt-to-income ratio, logistic regression models can predict the likelihood of default.

#### Library for Logistic Regression (in Python):
Scikit-learn also provides the `LogisticRegression` module, which is used to implement logistic regression models. We can start with logistic regression by importing this module from scikit-learn and use it to fit a logistic regression model to the data.

**Library Name**: Scikit-learn

**Link**: [Scikit-learn](https://scikit-learn.org/)

**Module to use**: `sklearn.linear_model.LogisticRegression`



**Dataset Example**: 
We can use the Iris dataset, which is a popular dataset for classification tasks and is included in scikit-learn. 

**Dataset Name**: Iris Dataset

We can load this dataset directly from scikit-learn using the following code:

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

<br></br>
*Example implementation from scratch and more explanation- https://github.com/edith141/linear-classification*



## Binary Classification

Binary classification is a supervised machine learning task where the goal is to classify input data into one of two possible classes. It is widely used in various applications such as spam detection, disease diagnosis, and fraud detection.

In binary classification, the algorithm learns a decision boundary that separates the two classes based on input features. Common algorithms used for binary classification include logistic regression, decision trees, random forests, support vector machines (SVM), and neural networks.

Binary classification involves predicting a binary outcome, typically represented as 0 or 1 (or negative and positive classes), based on input features. Evaluation metrics such as accuracy, precision are commonly used to assess the performance of binary classification models.

**For example,**
**Spam Detection**: Binary classification is commonly used in email filtering systems to classify incoming emails as either spam or non-spam based on features such as sender, subject, and content.

**Credit Card Fraud Detection**: Binary classification is used in finance to detect fraudulent transactions by classifying each transaction as either fraudulent or legitimate based on features such as transaction amount, location, and time.

#### Library for Binary Classification (in Python):
Scikit-learn provides a variety of algorithms and modules for binary classification tasks, like Neural Networks, including logistic regression which we discussed above.

**Library Name**: Scikit-learn

**Link**: [Scikit-learn](https://scikit-learn.org/)

**Module to use**: Various modules depending on the algorithm chosen- e.g., `sklearn.linear_model.LogisticRegression`, 
`sklearn.svm.SVC`, 
`sklearn.ensemble.RandomForestClassifier`, 
etc...


## Neural Networks

Neural networks are versatile models inspired by the human brain's structure and function. Composed of interconnected layers of nodes (artificial neurons), they excel in learning complex patterns and relationships from data and are used in applications like image recognition, image classification, natural language processing, etc.

Each node in a neural network can be thought of as a linear regression model in itself, comprising input data, weights, bias, and output. The output of a node is calculated as the weighted sum of its inputs plus a bias term, passed through an activation function to produce the final output.

  
![](https://skyengine.ai/se/images/blog/neuron_representation_cornell.jpeg)
    

 A simple formula representation will look like- 

`∑wixi + bias = w1x1 + w2x2 + w3x3 + bias`

`output = f(x) = 1 if ∑w1x1 + b>= 0; and 0 if ∑w1x1 + b < 0`

Where,
- **wi** = weight i, 
 - **xi** = input i,
 - **b** = bias
  
Neural networks leverage activation functions and adjustable weights to process input data and generate output predictions. Through iterative training processes, such as backpropagation, they adjust and optimize weights of node connections to minimize errors and improve performance.

#### Library for Neural Networks (in Python):
**Library Name**: TensorFlow

**Link**: [TensorFlow](https://www.tensorflow.org/)

**Module to use**: `tensorflow.keras.Sequential`

TensorFlow is a popular open-source machine learning framework developed by Google. It provides support for building and training neural networks, including tools for data preprocessing, model construction, and evaluation. We can start with neural networks in TensorFlow by using the `Sequential` model from the `tensorflow.keras` module.

**Dataset Example**: 
We can use the MNIST dataset, which is a standard dataset for handwritten digit classification and is often used as a benchmark in machine learning. 

**Dataset Name**: MNIST Dataset

We can load this dataset directly from TensorFlow like so:

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

<br></br>
*More on NN- https://www.ibm.com/cloud/learn/neural-networks*


### Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to effectively handle sequential data, where context is important, such as natural language processing, or image captioning. Unlike traditional feedforward neural networks, RNNs possess internal memory, allowing them to maintain context. RNNs take the output of a processing node and transmit the information back into the network. This results in theoretical "learning" and improvement of the network. 

This memory mechanism enables RNNs to process inputs of variable length and learn from past information to make predictions about future states. They are commonly used in tasks such as speech recognition, language translation, and sentiment analysis, where the order and context of input data are crucial. 

Learn more about RNNs- https://www.tensorflow.org/guide/keras/rnn


### ResNets (Residual Neural Networks)

With very deep neural networks, as the number of layers increase, all the weights receive update proportional to the derivative of the error function wrt to its current weight. In deep networks, this number reduces to very small (or to something not relevant) and weights will not be change effectively and it may completely stop the neural network from further training. This is called vanishing gradients. In other works,  we can say that the data is disappearing through the layers of the deep neural network due to very slow gradient descent. 

To counter this, ResNets were introduced, in which, a network is split into blocks and then passing the input into each block straight through to the next block, along with the residual output of the block minus the input to the block that is reintroduced, helping eliminate this problem. 

More about ResNets - https://en.wikipedia.org/wiki/Residual_neural_network


## Data Augmentaion and filtering, clenaning- 

### Augmentation- 
Data augmentation is used in machine learning to artificially increase the size and diversity of a training dataset by applying various transformations to the existing data. These transformations can include rotations, translations, scaling, flipping, jittering, wraping, etc. depending on the nature of the data and dataset. 

By augmenting the dataset, models can be trained on a more extensive and representative set of examples, which often leads to improved generalisation and performance.

### Filtering and Cleaning-
Filtering involves removing irrelevant or redundant data points from the dataset based on some specific criteria or conditions. This helps us to reduce noise and focus on relevant information. 

Cleaning, on the other hand, focuses on identifying and correcting errors, inconsistencies, or missing values in the dataset. This can be done by pruning the data, correcting values, removing outliers, etc. depending on the nature of the data. 

Both the processes make the dataset(s) more reliable and accurate for training machine learning models.



## Libraries 

1. **OpenCV**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning library. It has a range of functionalities for image and video processing, including feature detection, object recognition, and camera calibration. It is mainly used to image processing rather than machine learning in itself. For example, to convert images, augmenting images, etc. 

	More about OpenCV - https://opencv.org/
	Install OpenCV in python - `pip install opencv-python`
	
2. **Pandas**: Pandas is designed and mainly used for data manipulation and analysis. It provides data structures like DataFrame and Series, which we can use to efficiently handle and analyze structured data, and view it. Pandas simplifieIt also helps in tasks such as data cleaning, transformation (one-hot encoding, etc).

	More about Pandas - https://pandas.pydata.org/
	Install Pandas in python - `pip install pandas`

3. **Matplotlib**: Matplotlib is used for creating static, and interactive visualisations to represent data. It offers a wide range of plotting functions to generate various types of plots, including line plots, scatter plots, histograms, bar charts, etc, and also a combination of them.  

	More about Matplotlib - https://matplotlib.org/
	Install Matplotlib in Python - `pip install matplotlib`

4. **Plotly**: Plotly is similarily used for creating interactive visualisations, one key difference being it is web based. It also offers various types of interactive plots, including line charts, scatter plots, and 3D plots. It gives more interactive features allowing us to explore data dynamically, zoom in/out, and hover over data points for additional information, etc. 

	Learn more about Plotly - https://plotly.com/python/
	Install Plotly in Python - `pip install plotly`

5. **Beautiful Soup**: Beautiful Soup is used for web scraping and parsing HTML and XML documents. We can use it for extracting data from web pages, navigating the HTML tree, and searching for specific elements or attributes in the document. It is primarily used in web scraping and data collection tasks. 

	Learn more about Beautiful Soup - https://www.crummy.com/software/BeautifulSoup/bs4/doc/
	Install BS4 in Python - `pip install beautifulsoup4`


6. **Selenium**: Selenium is a popular open-source tool for automating web browser actions and used for data collection and scraping. It provides a WebDriver API that allows users to interact with web elements, simulate user actions (e.g., clicking buttons, filling forms). This makes it easier to scrape data from dynamic websites as compared to other similar scraping libraries.

	Learn more about Selenium - https://www.selenium.dev/
	Install Selenium in Python - `pip install selenium`


7. **Tensorflow**: TensorFlow is one of the most powerful and open-source machine learning frameworks developed by Google. It has a number of tools, libraries, and resources for building and deploying machine learning models at scale. It also offers extensive GPU support to run large scale models efficiently and reduce training times exponentially. TensorFlow supports both traditional machine learning algorithms and deep learning techniques, making it suitable for a wide range of applications, including image recognition, natural language processing, and reinforcement learning. 

	More about TensorFlow - https://www.tensorflow.org/
	Install TensorFlow in Python- `pip install tensorflow` OR `pip install tensorflow-cpu` for CPU only version.


8. **Keras**: Keras is a high-level neural networks/models API, written in Python and capable of running on top of TensorFlow, Theano, etc. It provides a user-friendly interface for building and training deep learning models. Keras has support for many kinds of networks - neural networks (supports both convolutional and recurrent networks), as well as their combinations. It also supports some pre built model definitions. It is widely used in machine learning domain. 

	Learn more about Keras - https://keras.io/
	Install Keras in Python - `pip install keras`
 
 

<br>
 *It is better to use a virtual env as discussed below to install/use the above packages.*
 
## Python Virtual Environments

### Venv
Python Virtual Environment (venv) provide isolated environments for Python projects, and ensures that each project has its own set of dependencies and Python interpreter without conflicting with other versions. This prevents dependency conflicts, allow for version compatibility, and simplify dependency management. 

To setup a new Virtual Environment (python):

1. **Create a new directory for your project (optional):**
	`mkdir my_project`
	`cd my_project`

2. **Create a new virtual environment:**
	`python3 -m venv my_env`
	This creates a new virtual environment named "my_env" in the current directory.

3. **Activating the Virtual Environment:**

	- On Windows:
	`my_env\Scripts\activate`

	- On macOS and Linux:
	`source my_env/bin/activate`


	The prompt should now indicate we are using a virtual environment now.

4. **Deactivating a Virtual Environment:**
	Now to deactivate the virtual environment and return to the global Python environment, run this command-

	`deactivate`


This enables us to create and manage isolated environments for our Python projects, ensuring that dependencies are installed separately for each project.


### Conda
Similarly, Conda is a package manager and environment manager that simplifies the installation and management of software packages and their dependencies. It enables users to create isolated environments, install packages from different channels, and manage dependencies across different platforms.

To install Miniconda (a minimal version of Anaconda) and create a new Conda environment:

1. **Install Miniconda**:
   - Download the appropriate Miniconda installer for your operating system from the official website: [Miniconda Downloads](https://docs.conda.io/en/latest/miniconda.html)
   - Run the installer and follow the on-screen instructions to install.

2. **Create a New Conda Environment**:
   - Use the following command to create a new Conda environment named "my_env":
     ```bash
     conda create --name my_env
     ```
   - You can specify the Python version for the environment by adding `python=x.x` to the command, where `x.x` is the desired Python version (e.g., `python=3.8`).

3. **Activate the Conda Environment**:
   - On Windows:
     ```bash
     conda activate my_env
     ```
   - On macOS and Linux:
     ```bash
     source activate my_env
     ```
4. **Deactivate a Conda Environment**:
```conda deactivate```
This will activate the `my_env` environment, and you can now install packages and work within this isolated environment.

## IDEs

1. VSCode
	VSCode is one of the most popular, open-source, and lightweight code editors. It also supports Python and ipynb (Interactive Python Notebooks /Jupyter Notebooks).
	https://code.visualstudio.com
	
2. PyCharm 
	PyCharm is another IDE specifically targeted towards python development, but it is not as lightweight as VSCode and offers a free and paid version.
	https://www.jetbrains.com/pycharm/
	
3. VIM
	VIM is a command promt, text-based editor that comes pre-installed with most Linux distrobutions. It is used mainly where a GUI is not available. We can quickly get started without any installation since it is already a part of most Linux Distros.
	https://www.vim.org# Python - ML & Analytics
	
4. Jupyter Lab
	This is a browser based, feature rich editor where we can code and execute line by line and see results in the very next line. This is especially useful for analyzing and viewing data where we need to do steps in parts.
	https://jupyter.org/install
	To install, we can simply run
	`pip install jupyterlab`
	And once done, we can run it using the following command-
	`jupyter lab`

