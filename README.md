<h1>PyTorch Image Classification: Cats vs. Dogs</h1>
<p>This repository contains a complete pipeline for training a deep learning model to classify images of cats and dogs. It uses the popular <strong>ResNet18</strong> architecture and the <strong>PyTorch</strong> framework. The project includes scripts for data preparation, model training, and model statistics.</p>
<p></p>
<hr>
<h2>📋 Table of Contents</h2>
<ul>
<li><a href="#overview">Project Overview</a></li>
<li><a href="#features">Features</a></li>
<li><a href="#setup">Setup and Installation</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#dependencies">Dependencies</a></li>
</ul>
<hr>
<h2 id="overview">📖 Project Overview</h2>
<p>The goal of this project is to demonstrate a standard workflow for a computer vision classification task. It uses the "Cats vs. Dogs" dataset from Kaggle and a pre-trained <strong>ResNet18</strong> model for transfer learning.</p>
<hr>
<h2 id="features">✨ Features</h2>
<ul>
<li><strong>Automated Data Download</strong>: Uses the <code>kagglehub</code> library to automatically download and prepare the dataset.</li>
<li><strong>Transfer Learning</strong>: Leverages a pre-trained ResNet18 model for high accuracy with minimal training.</li>
<li><strong>Model Training &amp; Saving</strong>: A clear script to train the model and save the best-performing weights.</li>
<li><strong>Organized Codebase</strong>: Scripts are separated by function (data prep, training) for clarity.</li>
</ul>

<hr>

<h2 id="setup">💻 Setup and Installation</h2>
<p>To get started, clone the repository and install the required dependencies.</p>
<p><strong>1. Clone the repository:</strong></p>
<pre><code>git clone https://github.com/Cwalt2/PyTorch-classification.git
cd PyTorch-classification</code></pre>
<p><strong>2. Create a virtual environment (recommended):</strong></p>
<pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`</code></pre>
<p><strong>3. Install dependencies:</strong></p>
<p>Make sure you have a <code>requirements.txt</code> file with the required packages. Then, install them:</p>
<pre><code>pip install -r requirements.txt</code></pre>
<p><strong>4. Kaggle API Authentication:</strong></p>
<p>This project uses <code>kagglehub</code> to download the dataset. You will need to have your Kaggle API credentials set up by placing your <code>kaggle.json</code> file in the <code>~/.kaggle/</code> directory.</p>

<hr>

<h2 id="usage">🚀 Usage</h2>
<p>The project is designed to be run in three sequential steps.</p>
<h3>1. Data Preparation</h3>
<p>First, run the <code>data-prep.py</code> script. This will download the dataset and organize it into <code>train</code> and <code>val</code> directories.</p>
<pre><code>python data-prep.py</code></pre>
<h3>2. Model Training</h3>
<p>Next, run the <code>train.py</code> script to start training the ResNet18 model on the prepared data. This will save the best model weights to a file named <code>cat_dog_model.pth</code>.</p>
<pre><code>python train.py</code></pre>
<h3>3. Model Statistics and Testing</h3>
<p>Running the <code>test.py</code> script will run the model in the <code>path</code> variable and the <code>model-stats.py</code> script will analyze the model and output stats like accuracy</p>
<pre><code>python model-stats.py</code></pre>
<pre><code>python test.py</code></pre>
<p></p>

<hr>

<h2 id="dependencies">📦 Dependencies</h2>
<ul>
<li><strong>torch &amp; torchvision</strong></li>
<li><strong>kagglehub</strong></li>
<li><strong>numpy</strong></li>
<li><strong>Pillow</strong></li>
</ul>