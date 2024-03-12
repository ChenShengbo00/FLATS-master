## Local Perturbation-Based Black-Box Federated Learning Attack for Time Series Classification

### Abstract: 
The widespread adoption of intelligent machines and sensors has generated vast amounts of time series data, leading to the 
increasing use of neural networks in time series classification. Federated learning has emerged as a promising machine learning
paradigm that reduces the risk of user privacy leakage. However, federated learning is vulnerable to backdoor attacks, which
pose significant security threats. Furthermore, existing unrealistic white-box methods for attacking time series result in insufficient
adaptation and inferior stealthiness. To overcome these limitations, this paper proposes a gradient-free black-box method called
local perturbation-based backdoor Federated Learning Attack for Time Series classification (FLATS). The attack is formulated
as a constrained optimization problem and is solved using a differential evolution algorithm, without requiring any knowledge
of the internal architecture of the target model. In addition, the proposed method considers the time series shapelet interval as a
local perturbation range and adopts a soft target poisoning approach to minimize the difference between the attacker model and
the benign model. Experimental results demonstrate that our proposed method can effectively attack federated learning time series
classification models with potential security issues while generating imperceptible poisoned samples that can evade various defence
methods

### Download Dataset
Before you start the experiment, you need to prepare the dataset, you can download the data from:
"https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"

Then you need to divide the dataset, you can employ "stratify_attack_data(dataset name)"
to sample the attack dataset

### Running Experients

#### Tested stable depdencises
* conda 4.11.0
* python 3.8.12
* pandas  1.4.1
* torchvision   0.2.2
* scikit-learn   1.0.2

#### The main script is "tsadv.py"

python3.8 tsadv.py 
--lr=0.0005
--gamma=0.997
--local_train_period=1
--eps=1
--prox_attack=False
--cuda
--num_nets=100
--fl_round=300
--part_nets_per_round=20
--dataset=MelbournePedestrian
--attacker_pool_size=5
--defense_method=ndc
--model=resnet
--attackmathed=Attacker
--fl_mode=fixed-pool

attackmathed = Attacker or AttackerRandShape or AttackerRandAll or AttackerOnepoint

fl_mode=fixed-pool or fixed-freq, and while fl_mode=fixed-freq the attacker_pool_size=1
