
# Python - ML & Analytics


### Machine Learning Basic Termninology-

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

### Linear Regression

Linear regression is a supervised machine learning algorithm used to predict continuous values, such as sales figures or housing prices. It analyzes the relationship between two or more variables, where one variable, known as the dependent variable, is predicted based on the values of other variables, called independent variables.

The fundamental concept behind linear regression is to model the linear relationship between the independent and dependent variables. This relationship is represented by a straight line, known as the regression line, which shows how the value of the dependent variable changes as the independent variable(s) vary. The equation of the regression line is typically expressed as:

**`y = wx + b`**

Where:
- **X**: The independent variable.
- **Y**: The dependent variable that the model will predict.
- **Weight (w)**: The coefficient for the independent variable X. In machine learning terminology, these coefficients are referred to as weights.
- **Bias (b)**: The Y-intercept, which offsets all predicted values of Y. In machine learning, this is known as the bias term.

Linear regression is a versatile tool used in various fields, including economics, finance, and healthcare, for modeling relationships and making predictions based on observed data.

### Linear Regression- 

This is a (supervised) machine learning algo, basically used to predict continuous value(s) like sale figures, etc. Linear regression analysis is **used to predict the value of a variable based on the value of another variable**. The variable you want to predict is called the **dependent variable**. The variable you are using to predict the other variable's value is called the **independent variable.**

It is a statistical approach that represents the linear relationship between two or more variables, either dependent or independent, hence called Linear Regression. It shows the value of the dependent variable changes with respect to the independent variable, and the slope of this graph is called as Line of Regression.

The basic LR is of the form  **`y = wx + b`**  

_**X**_- The independent variable.

_**Y**_- The dependent variable (the model will predict Y values).

_**Weight (w)-**_  The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as  _weights_.

_**Bias (b)**_  -The Y-intercept. In ML, we'll call this Bias. This essentially offsets all the predicted values of Y.

Example implementation and more explanation - https://github.com/edith141/LinearReg-scratch
Implemented activation function, cost function, gradient descent, etc.

### Logistic Regression
This is used to classify (or "predict classification") items/data to a class from the given possible classes. For example, in binary classification, i.e. we predict if a data/item belongs to a given class or not. One simple way to see this is as an answer to the question "Does this data item belong to this class?" The answer could either be True or False (binary) based on probability of that data belonging to a particular class. 

Now, as we did in LR, we try and fit a line to the given data points, but in the data is not linearly separable, we use other functions.

For example, we can use a  **logistic**  (sigmoid) function. This also uses an equation as its representation. The input value is combined with the coefficients (wts and bias) to predict an o/p- y.

Now, in terms of input "x" and weight "w", we can write this as:
$$ y =g(w.x) = \frac{1}{1 +e^(-w.x^)} $$

_**X**_- The independent variable.

_**Y**_- The dependent variable (the model will predict Y values).

_**Weight (w)-**_  The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as  _weights_.


Here, one key difference here to note is the o/p is either 0 or 1 (binary).

 Example Implementation and explanation- https://github.com/edith141/linear-classification
 Implemented activation (sigmoid) function, cost function, gradient descent, etc. in the above example.
 

### Binary Classification- 
Binary classification is a supervised machine learning algo where we classify a data to one of the two possible classes using suitable models. There are many applications fir binary applications like- email classification (spam or not), fraud detection for transactions (fraud or not), image classification (cat or not a cat), or medical (malignant or not), etc. 

We can use neural networs, logistic regression, etc for binary classification as shown above, where we use logistic regression for classification. 


### Neural Networks- 
Neural networks are versatile models inspired by the human brain's structure and function. Composed of interconnected layers of nodes, (artificial neurons), they excel in learning complex patterns and relationships from data and are used in applications like image recognition, image classification, natural language processing, etc. Each node here can be thought of as a linear regression model in itself, composed of input data, weight, bias, and output. A simple formula representation will look like- 

`∑wixi + bias = w1x1 + w2x2 + w3x3 + bias`

`output = f(x) = 1 if ∑w1x1 + b>= 0; and 0 if ∑w1x1 + b < 0`

*Where
wi = weight i, 
xi = input i,
b = bias*
 
They leverage activation functions and adjustable weights to process input data and generate output predictions. In iterative training processes, through backpropagation, they adjust and optimize weights of node connections to minimise errors.  

More on NN- https://www.ibm.com/cloud/learn/neural-networks

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


### Linear Regression- 

This is a (supervised) machine learning algo, basically used to predict continuous value(s) like sale figures, etc. Linear regression analysis is **used to predict the value of a variable based on the value of another variable**. The variable you want to predict is called the **dependent variable**. The variable you are using to predict the other variable's value is called the **independent variable.**

The basic LR is of the form  **`y = wx + b`**  

_**X**_- The independent variable.

_**Y**_- The dependent variable (the model will predict Y values).

_**Weight (w)-**_  The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as  _weights_.

_**Bias (b)**_  -The Y-intercept. In ML, we'll call this Bias. This essentially offsets all the predicted values of Y.

Example implementation and more explanation - https://github.com/edith141/LinearReg-scratch
Implemented activation function, cost function, gradient descent, etc.

### Logistic Regression
This is used to classify (or "predict classification") items/data to a class from the given possible classes. For example, in binary classification, i.e. we predict if a data/item belongs to a given class or not. One simple way to see this is as an answer to the question "Does this data item belong to this class?" The answer could either be True or False (binary) based on probability of that data belonging to a particular class. 

Now, as we did in LR, we try and fit a line to the given data points, but in the data is not linearly separable, we use other functions.

For example, we can use a  **logistic**  (sigmoid) function. This also uses an equation as its representation. The input value is combined with the coefficients (wts and bias) to predict an o/p- y.

Now, in terms of input "x" and weight "w", we can write this as:
$$ y =g(w.x) = \frac{1}{1 +e^(-w.x^)} $$

_**X**_- The independent variable.

_**Y**_- The dependent variable (the model will predict Y values).

_**Weight (w)-**_  The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as  _weights_.


Here, one key difference here to note is the o/p is either 0 or 1 (binary).

 Example Implementation and explanation- https://github.com/edith141/linear-classification
 Implemented activation (sigmoid) function, cost function, gradient descent, etc. in the above example.
 

### Binary Classification- 
Binary classification is a supervised machine learning algo where we classify a data to one of the two possible classes using suitable models. There are many applications fir binary applications like- email classification (spam or not), fraud detection for transactions (fraud or not), image classification (cat or not a cat), or medical (malignant or not), etc. 

We can use neural networs, logistic regression, etc for binary classification as shown above, where we use logistic regression for classification. 


### Neural Networks- 
Neural networks are versatile models inspired by the human brain's structure and function. Composed of interconnected layers of nodes, (artificial neurons), they excel in learning complex patterns and relationships from data and are used in applications like image recognition, image classification, natural language processing, etc. Each node here can be thought of as a linear regression model in itself, composed of input data, weight, bias, and output. A simple formula representation will look like- 

`∑wixi + bias = w1x1 + w2x2 + w3x3 + bias`

`output = f(x) = 1 if ∑w1x1 + b>= 0; and 0 if ∑w1x1 + b < 0`

*Where
wi = weight i, 
xi = input i,
b = bias*
 
They leverage activation functions and adjustable weights to process input data and generate output predictions. In iterative training processes, through backpropagation, they adjust and optimize weights of node connections to minimise errors.  

More on NN- https://www.ibm.com/cloud/learn/neural-networks

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
	https://www.vim.org
