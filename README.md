

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

Unsupervised learning finds applications in various fields where labeled data is scarce or unavailable. It is commonly used in:

1. **Clustering**: Grouping similar data points together based on their inherent characteristics. For example, clustering customers based on purchasing behavior to identify market segments.

2. **Recommendation Systems**: Generating personalized recommendations for users based on their preferences and behaviors, commonly used in online streaming services, e-commerce platforms, and social media platforms.

3. **Natural Language Processing (NLP)**: Uncovering semantic relationships and structures in text data for tasks like document summarisation, sentiment analysis, and language translation.



## Linear Regression

Linear regression is a supervised machine learning algorithm used to predict continuous values, such as sales figures or housing prices. It analyzes the relationship between two or more variables, where one variable, known as the **dependent variable**, is predicted based on the values of other variables, called **independent variables**.

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

**Dataset Example**: We can use the California Housing dataset, which is already included in scikit-learn and provides information about housing prices in California. 

**Dataset Name**: California Housing Dataset

We can load this dataset directly from scikit-learn using the following module: 
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```

<br></br>

*Example implementation from scratch and more explanation -* https://github.com/edith141/LinearReg-scratch


## Multiple Linear Regression

Multiple linear regression is a supervised machine learning technique used to predict continuous values by analyzing the relationship between multiple independent variables and a single dependent variable. It extends the concept of linear regression by considering more than one predictor variable.

The primary objective of multiple linear regression is to model the linear relationship between the independent variables (features) and the dependent variable. This relationship is represented by a linear equation of the form:

**`y = w1*x1 + w2*x2 + ... + wn*xn + b`**

Where:
- **y**: The dependent variable being predicted.
- **x1, x2, ..., xn**: Independent variables (features) influencing the dependent variable.
- **w1, w2, ..., wn**: Coefficients (weights) representing the impact of each independent variable on the dependent variable.
- **b**: Bias or intercept term.

Multiple linear regression aims to find the optimal values of the coefficients (weights) and the bias term to minimize the difference between the predicted values and the actual values of the dependent variable.

Multiple linear regression is widely used in various domains such as finance, marketing, and social sciences for tasks including sales forecasting, risk assessment, and demand prediction.

### Library for Multiple Linear Regression (in Python):

**Library Name**: Scikit-learn

**Link**: [Scikit-learn](https://scikit-learn.org/)

**Module to use**: `sklearn.linear_model.LinearRegression`

As done in Linear Regression, we can use the same LinearRegression module in sklearn for multiple linear regression.

**Dataset Example**: For this example, we can use the same California Housing dataset in scikit-learn that provides information about housing prices in different regions of California. We can load this dataset using the following code:

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```

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

---
**K-Means Clustering**

K-Means Clustering is an unsupervised machine learning algorithm used for partitioning data into clusters based on similarity. It aims to group data points into k clusters where each data point belongs to the cluster with the nearest mean.

**Algorithm Overview:**

1. **Initialization:** Choose k initial cluster centroids randomly from the data points.

2. **Assignment:** Assign each data point to the nearest cluster centroid based on a distance metric, such as Euclidean distance.

3. **Update Centroids:** Update the cluster centroids by computing the mean of all data points assigned to each cluster.

4. **Repeat:** Repeat steps 2 and 3 until convergence, where the cluster assignments and centroids no longer change significantly.

**Mathematical Representation:**

In K-Means Clustering, the objective is to minimize the within-cluster sum of squares, which can be mathematically represented as:

$$
\underset{C}{\text{argmin}} \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$


Where:

-   C is the set of clusters.
-   Ci​ is the i-th cluster.
-   μi is the centroid of the i-th cluster.

**Applications:**

1. **Customer Segmentation:** K-Means Clustering can segment customers based on their purchasing behavior or demographic information for targeted marketing strategies.

2. **Image Compression:** K-Means Clustering can reduce the number of colors in an image by clustering similar pixel values together, resulting in compressed images.

3. **Document Clustering:** K-Means Clustering can cluster text documents based on their content for topic modeling or document organization.

4. **Anomaly Detection:** K-Means Clustering can identify outliers or anomalous data points by assigning them to a separate cluster.

**Library and Module:**

- Library Name: Scikit-learn
- Module to Use: `sklearn.cluster.KMeans`

---

**Support Vector Machines (SVM)**

Support Vector Machines (SVM) is a versatile supervised machine learning algorithm used for classification, regression, and outlier detection tasks. It finds the optimal hyperplane that best separates data points into different classes or fits a nonlinear function to the data.

**Algorithm Overview:**

1. **Maximizing Margin:** SVM aims to maximize the margin between classes, which is the distance between the hyperplane and the nearest data points from each class, also known as support vectors.

2. **Kernel Trick:** SVM can efficiently handle non-linear decision boundaries by mapping input features into a higher-dimensional space using kernel functions.

3. **Regularization:** SVM includes a regularization parameter (C) to balance between maximizing the margin and minimizing classification errors on the training data.

**Mathematical Representation:**

In classification scenarios, the decision boundary (hyperplane) can be represented as:

$$w^tx+b=0$$

Where:
- $w$ is the weight vector.
- $x$ is the input feature vector.
- $b$ is the bias term.

The classification decision is made based on the sign of \( w^Tx + b \).

**Applications:**

1. **Multi-class Classification:** SVMs can be extended to handle multi-class classification problems by using strategies like one-vs-rest (OvR) or one-vs-one (OvO) classification.

2. **Regression:** SVMs can be used for regression tasks by fitting a hyperplane to the data that best predicts continuous target variables.

3. **Outlier Detection:** SVMs can detect outliers in data by identifying data points that lie farthest from the decision boundary.

4. **Anomaly Detection:** SVMs can identify anomalies or unusual patterns in data, such as fraudulent transactions in financial transactions or defects in manufacturing processes.

**Library and Module:**

- Library Name: Scikit-learn
- Module to Use: `sklearn.svm.SVC` for classification, `sklearn.svm.SVR` for regression, and `sklearn.svm.OneClassSVM` for outlier detection.

**Example Dataset:**

- Various datasets are available in Scikit-learn or can be sourced from real-world applications for experimentation.

**Use Cases:**

1. **Handwritten Digit Recognition:** SVMs can classify handwritten digits into multiple classes (0-9) based on pixel intensities in images.

2. **Facial Expression Recognition:** SVMs can recognize facial expressions, such as happy, sad, angry, or neutral, from images or video frames.

3. **Stock Price Prediction:** SVMs can predict stock prices by analyzing historical stock data and identifying patterns that indicate potential price movements.

4. **Text Categorization:** SVMs can classify text documents into multiple categories, such as news articles, emails, or product reviews.

5. **Fault Diagnosis:** SVMs can detect faults or anomalies in industrial machinery based on sensor data, helping to prevent breakdowns and improve maintenance scheduling.


---


## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the original data's variance. It achieves this by identifying the principal components, which are orthogonal vectors that capture the directions of maximum variance in the data.

final sklearn
For example, in the above image, the red values are the values predicted by the Linear Regression model and the green values are the original data values (input). The red line is the regression line that can be used to predict future values.

**Algorithm Overview:**

1. **Standardization:** Standardize the features by subtracting the mean and dividing by the standard deviation.

2. **Compute Covariance Matrix:** Calculate the covariance matrix of the standardized data.

3. **Eigenvalue Decomposition:** Compute the eigenvectors and eigenvalues of the covariance matrix.

4. **Select Principal Components:** Sort the eigenvectors based on their corresponding eigenvalues and select the top \( k \) eigenvectors as the principal components.

5. **Transform Data:** Project the original data onto the selected principal components to obtain the lower-dimensional representation.

**Mathematical Representation:**

The covariance matrix \( \Sigma \) of the standardized data matrix \( X \) is computed as:

$$
\Sigma = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X})
$$

The eigenvectors \( \mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_d \) and corresponding eigenvalues \( \lambda_1, \lambda_2, ..., \lambda_d \) of \( \Sigma \) satisfy the equation:

$$
\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i
$$

After sorting the eigenvectors based on their corresponding eigenvalues in descending order, the top \( k \) eigenvectors are selected as the principal components \( \mathbf{V} \).

The lower-dimensional representation of the data \( X' \) is obtained by projecting the original data onto the principal components:

$$
X' = X \cdot \mathbf{V}
$$

**Applications:**

1. **Dimensionality Reduction:** PCA is widely used for reducing the dimensionality of high-dimensional datasets while preserving most of the variance, making it easier to visualize and analyze data.

2. **Feature Extraction:** PCA can be used to extract the most important features from the data, leading to better performance in downstream machine learning tasks.

3. **Noise Reduction:** PCA can help in removing noise and redundant information from data, improving the signal-to-noise ratio.

4. **Image Compression:** PCA is used in image processing for compressing images by representing them in a lower-dimensional space without significant loss of information.


### Implementation Example in Python

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
```

This demonstrates how to perform PCA using the `PCA` class from scikit-learn. It reduces the dimensionality of the Iris dataset to 2 dimensions.

---

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


## Recurrent Neural Networks (RNN)

Recurrent Neural Networks (RNNs) are a class of neural networks designed to efficiently process sequential data, such as time-series data or natural language text. Unlike feedforward neural networks, which process each input independently, RNNs maintain an internal state (memory) that captures information about previous inputs. This recurrent connection allows RNNs to exhibit dynamic temporal behaviour and handle sequences of arbitrary length.

RNNs are widely used in various applications, including natural language processing (NLP), speech recognition, sentiment analysis, and time-series forecasting. They excel in tasks that require capturing long-term dependencies and context information from sequential data.

One of the key components of an RNN is the hidden state, which represents the network's memory of previous inputs. At each time step, the hidden state is updated based on the current input and the previous hidden state, allowing the network to retain information over time.

#### Library for Recurrent Neural Networks (in Python):
**Library Name**: TensorFlow

**Link**: [TensorFlow](https://www.tensorflow.org/)

**Module to use**: `tensorflow.keras.layers.SimpleRNN` or `tensorflow.keras.layers.LSTM` or `tensorflow.keras.layers.GRU`

TensorFlow provides various layers for implementing recurrent neural networks, including SimpleRNN, LSTM (Long Short-Term Memory), etc. These layers can be used to construct RNN architectures for different tasks and data types. We can start with RNNs in TensorFlow by using these layers within the `tensorflow.keras` module.

**Dataset Example**: 
You can use the IMDb dataset, which is a popular dataset for sentiment analysis of movie reviews. This dataset contains text sequences of movie reviews along with their corresponding labels (positive or negative sentiment).

**Dataset Name**: IMDb Dataset

You can load this dataset directly from TensorFlow using the following code:

```python
import tensorflow as tf
imdb = tf.keras.datasets.imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()
```

<br></br>
*More about RNNs- https://www.tensorflow.org/guide/keras/rnn*


## Residual Neural Networks (ResNets)

Residual Neural Networks (ResNets) are a type of deep neural network architecture designed to address the vanishing gradient problem in very deep networks. Traditional deep neural networks often suffer from degradation (increased training error) as the network depth/number of layers increase, making it challenging to train extremely deep models.

ResNets introduce skip connections, also known as residual connections, that bypass one or more layers in the network. These skip connections allow the gradient to flow more directly through the network during training, mitigating the vanishing gradient problem and enabling the training of very deep networks.

ResNets are widely used in computer vision tasks, such as image classification, object detection, and image segmentation, where deep convolutional neural networks are required to learn complex hierarchical representations from visual data.

#### Library for Residual Neural Networks (in Python):
TensorFlow provides pre-trained ResNet models, including ResNet50, which can be used for various computer vision tasks out of the box. We can use these pre-trained models directly or fine-tune them to test/use them.

**Library Name**: TensorFlow

**Link**: [TensorFlow](https://www.tensorflow.org/)

**Module to use**: `tensorflow.keras.applications.ResNet50`


**Dataset Example**: 
We can use the CIFAR-10 dataset, which is a popular benchmark dataset for image classification tasks. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

**Dataset Name**: CIFAR-10 Dataset

We can load this dataset directly from TensorFlow using the following code:

```python
import tensorflow as tf
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

*More about ResNets - https://en.wikipedia.org/wiki/Residual_neural_network*


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

