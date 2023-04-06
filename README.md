
# Computer Science Master in University of Bonn

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/uni_bonn.jpg?raw=true">

The [University of Bonn](https://en.wikipedia.org/wiki/University_of_Bonn) is one of the largest research-based universities in the world. The THE ranking is [89](https://www.timeshighereducation.com/world-university-rankings/2023/world-ranking) and the QS ranking is [201](https://www.topuniversities.com/university-rankings/world-university-rankings/2023) in the world in 2023. 

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/desk.JPEG?raw=true">

Being the best university in Germany in the field of mathematics, Uni of Bonn has also adjusted its computer science department to have a solid mathematical foundation. For this reason, many courses are built on combining heavy mathematical theory and practical coding skills, and the language of instruction is 100% English.

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/bonn_diploma.JPEG?raw=true" width="80%" height="70%">

## Lectures, Thesis, Labs, Seminars

I took a total of 120 credits with a total of 14 courses and thesis+seminar from three different fields. All lecture details can be seen in files here with a summary of the content of each one of them. The programming languages in classes are mostly Python and C++. The transcript and diploma supplement can be seen [here](https://github.com/cuneyterem8/uni_bonn_background/tree/main/diplomas)


intelligence systems: [artificial life](https://github.com/cuneyterem8/uni_bonn_background/tree/main/artificial_life), [machine learning](https://github.com/cuneyterem8/uni_bonn_background/tree/main/machine_learning), [neural networks](https://github.com/cuneyterem8/uni_bonn_background/tree/main/neural_networks), [robot learning](https://github.com/cuneyterem8/uni_bonn_background/tree/main/robot_learning), [cognitive robotics](https://github.com/cuneyterem8/uni_bonn_background/tree/main/cognitive_robotics), [cognitive robotics lab](https://github.com/cuneyterem8/uni_bonn_background/tree/main/cognitive_robotics_lab), [humanoid robots lab](https://github.com/cuneyterem8/uni_bonn_background/tree/main/humanoid_robots_lab), [humanoid robots seminar](https://github.com/cuneyterem8/uni_bonn_background/tree/main/humanoid_robots_seminar)

computer-vision-audio: [deep learning for visual recognition](https://github.com/cuneyterem8/uni_bonn_background/tree/main/deep_learning_for_visual_recognition), [audio signal processing](https://github.com/cuneyterem8/uni_bonn_background/tree/main/audio_signal_processing)

information systems: [thesis in usable security and privacy](https://github.com/cuneyterem8/uni_bonn_background/tree/main/thesis_usable_security_privacy), [usable security and privacy](https://github.com/cuneyterem8/uni_bonn_background/tree/main/usable_security_privacy), [it security](https://github.com/cuneyterem8/uni_bonn_background/tree/main/it_security), [sensor data fusion](https://github.com/cuneyterem8/uni_bonn_background/tree/main/sensor_data_fusion), [mobile communication](https://github.com/cuneyterem8/uni_bonn_background/tree/main/mobile_communication)

only exercises (not exams due to lack of preparation): [biomedical data science](https://github.com/cuneyterem8/uni_bonn_background/tree/main/biomedical_data_science_hw_only), [humanoid robots](https://github.com/cuneyterem8/uni_bonn_background/tree/main/humanoid_robots_hw_only), [foundation of graphics](https://github.com/cuneyterem8/uni_bonn_background/tree/main/foundation_of_graphics_hw_only), [network security](https://github.com/cuneyterem8/uni_bonn_background/tree/main/network_security_hw_only)


### Example images from labs and thesis 

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/humanoid_robots_lab/turtlebot2_map.gif?raw=true" width="60%" height="50%">

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/thesis_usable_security_privacy/thesis_website.gif?raw=true">


### Implementation example from Deep learning class

```python
import torch
import numpy as np
from tqdm import tqdm 
from time import sleep 
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
```
```python
train_dataset = datasets.CIFAR10(root = "cifar/", train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = "cifar/", train = False, download = True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = True, pin_memory=True)
```
```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.bn = nn.BatchNorm1d(512, affine=True)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 32*32*3)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.relu(self.bn(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def forward_test(self, x):
        x = x.reshape(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
def train(model, optimizer, dataloader, batch:bool):
    for x, y in dataloader:
        x, y = x.to('cuda:0'), y.to('cuda:0')
        optimizer.zero_grad()
        if batch == True:
            prediction = model.forward(x)
        else: 
            prediction = model.forward_test(x)
        loss = nn.CrossEntropyLoss()
        output_loss = loss(prediction, y.to(torch.long))
        output_loss.backward()
        optimizer.step()
    return output_loss

def accuracy(model, dataloader):
    hits = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to('cuda:0'), y.to('cuda:0')
            prediction = model.forward_test(x)
            prediction = torch.argmax(prediction, dim=1)
            hits += (prediction == y).count_nonzero()
    acc = hits / len(dataloader.dataset)
    return acc
```
```python
def batch_norm(model_factory, optimizer_factory, dataloader, epochs):
    losses = torch.zeros(epochs)
    accuracies = torch.zeros(epochs)
    model = model_factory.to('cuda:0')
    optimizer = optimizer_factory
    for epoch in tqdm(range(epochs)):
        losses[epoch] = train(model, optimizer, dataloader, True)
        accuracies[epoch] = accuracy(model, test_loader)
        sleep(0.1)
    return losses, accuracies

def baseline_nobatch(model_factory, optimizer_factory, dataloader, epochs):
    losses = torch.zeros(epochs)
    accuracies = torch.zeros(epochs)
    model = model_factory.to('cuda:0')
    optimizer = optimizer_factory
    for epoch in tqdm(range(epochs)):
        losses[epoch] = train(model, optimizer, dataloader, False)
        accuracies[epoch] = accuracy(model, test_loader)
        sleep(0.1)
    return losses, accuracies
```
```python
model = Network()
optimizer  = optim.Adam(model.parameters(), lr= 0.005, betas= (0.9, 0.95))
batch_losses, batch_accuracies = batch_norm(model, optimizer, train_loader, 25)

model1 = Network()
optimizer1  = optim.Adam(model1.parameters(), lr= 0.005, betas= (0.9, 0.95))
baseline_losses, baseline_accuracies = baseline_nobatch(model1, optimizer1, train_loader, 25)

def plotting_losses(batch, base):
    plt.plot(batch.detach(), 'r', label="With Batch Norm")
    plt.plot(base.detach(), 'k', label="Withouot Batch Norm")
    plt.legend()
    plt.show()
	
def plotting_accuracies(batch, base):
    plt.plot(batch, 'r', label= "With batch norm")
    plt.plot(base, 'k', label= "without batch norm")
    plt.legend()
    plt.show()

plotting_losses(batch_losses[:5], baseline_losses[:5])
plotting_accuracies(batch_accuracies, baseline_accuracies)
```

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/deep_learning_for_visual_recognition/output1.png?raw=true" width="60%" height="60%">

> output 1

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/deep_learning_for_visual_recognition/output2.png?raw=true" width="60%" height="60%">

> output 2


### Theory example from Machine learning class

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/machine_learning/image1.png?raw=true" width="80%" height="70%">

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/machine_learning/image2.png?raw=true" width="80%" height="70%">


# Certificate Projects

[Machine Learning A-Z](https://github.com/cuneyterem8/certificate_projects/tree/main/machine_learning_sds_udemy) in Python Udemy

Part 1 - Data Preprocessing \
Part 2 - Regression: Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, SVR, Decision Tree Regression, Random Forest Regression \
Part 3 - Classification: Logistic Regression, K-NN, SVM, Kernel SVM, Naive Bayes, Decision Tree Classification, Random Forest Classification \
Part 4 - Clustering: K-Means, Hierarchical Clustering \
Part 5 - Association Rule Learning: Apriori, Eclat \
Part 6 - Reinforcement Learning: Upper Confidence Bound, Thompson Sampling \
Part 7 - Natural Language Processing: Bag-of-words model and algorithms for NLP \
Part 8 - Deep Learning: Artificial Neural Networks, Convolutional Neural Networks \
Part 9 - Dimensionality Reduction: PCA, LDA, Kernel PCA \
Part 10 - Model Selection & Boosting: k-fold Cross Validation, Parameter Tuning, Grid Search, XGBoost

<img src="https://github.com/cuneyterem8/certificate_projects/blob/main/machine_learning_sds_udemy/ml_udemy.jpg?raw=true" width="80%" height="80%">

[Neural Networks and Deep Learning](https://github.com/cuneyterem8/certificate_projects/tree/main/neural_networks_deep_learning_dai_coursera) in Python Coursera

1- Introduction to Deep Learning: covers the analysis of the significant trends that lead to the growth of deep learning and provides examples of its current applications. \
2- Neural Networks Basics: teaches students how to approach a machine learning problem with a neural network perspective and how to use vectorization to accelerate their models. \
3- Shallow Neural Networks: guides students in building a neural network with one hidden layer using forward propagation and backpropagation. \
4- Deep Neural Networks: focuses on understanding the crucial computations that support deep learning, and how to use them to create and train deep neural networks for computer vision tasks.

<img src="https://github.com/cuneyterem8/certificate_projects/blob/main/neural_networks_deep_learning_dai_coursera/dl1_coursera.png?raw=true" width="80%" height="80%">


# Java Software Engineer for 1 year

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/TEB_LOGO.png" width="40%" height="30%">

I worked in the campaign software unit within the CRM software of TEB bank for almost 1 year. [Turkish Economy Bank](https://en.wikipedia.org/wiki/T%C3%BCrk_Ekonomi_Bankas%C4%B1) is one of the top 10 largest and oldest banks in Turkey, headquartered in Istanbul.

During this period, I worked on tasks such as developing in-bank software with Java language and evam tool (in-bank software), updating existing codes, and designing applications for the needs of business units.

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/teb_doc.jpeg" width="80%" height="70%">

# Computer Engineering Bachelor in Bilkent University

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/uni_bilkent.jpg">

The [Bilkent University](https://en.wikipedia.org/wiki/Bilkent_University) is one of the top 5 research-based universities in Turkey, and it is located in Ankara. The THE/QS rankings are 500 in the world in 2018 to 2023.

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/bilkent_diploma.JPEG">

The courses at the university highlight the combination of theory and practice, as well as algorithmic thinking, and the language of instruction is 100% English. During the university, I took many basic computer engineering courses as well as elective courses. The projects can be seen in my [previous github account](https://github.com/cuneyteremcs). The programming languages in classes are mostly Java and C++. The transcript and diploma supplements can be seen [here](https://github.com/cuneyterem8/uni_bonn_background/tree/main/diplomas). 

main classes: [final project: Helthscope App](https://github.com/cuneyteremcs/HealthScope), software engineering project management, systems analysis and design, principles of engineering management, software product line engineering, artificial intelligence

compulsory classes: algorithms, programming, object-oriented software engineering, data structures, operating systems, database systems, calculus, linear algebra, probability and statistics

internships: [System Administrator at Turkey Is Bank](https://github.com/cuneyteremcs/IsBank-Summer-Internship) (1.5 month), [Android Java Developer at Mia Technology](https://github.com/cuneyteremcs/MiaCamProject) (1 month)

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/bilkent_transcript_1.JPEG" width="80%" height="70%">

<img src="https://github.com/cuneyterem8/uni_bonn_background/blob/main/diplomas/bilkent_transcript_2.JPEG" width="80%" height="70%">
