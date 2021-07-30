# UPMSP_single

## Table of contents

+ [General info](#general-info)
+ [Requirements](#requirements)
+ [How to Use](#how-to-use)



## General info

The program __UPMSP__ is the __reinforcement learning based model for solving Unrelated Parallel Machine Scheduling Problem.__

__keywords__ : UPMSP, DDQN, Descrete Event Simulation(DES)



##  Requirements

This program requires the following modules:

+ python=3.6
+ tensorflow-gpu==2.1.0 (install gpu version of tensorflow module)
+ tensorflow==2.1.0  (install cpu version of tensorflow module)
+ scipy==1.4.1
+ simpy==4.0.1
+ openpyxl
+ xlrd
+ numpy==1.19.5
+ pandas
+ matplotlib
+ PyCharm under version of 2021.1





## How to Use



### Job shop scheduling problem settings



Weight(for tardiness) and process time for each job type and machine were set fixed, but parts' __job types__ __and __ __inter arrival time(IAT)__ __were__ __randomly generated in each training episode__



+ __Parallel_Machine.py__ > <class> __JobShop__ > <function>__ __init__ __()

  + Set environment parameters under __Parallel_Machine.py__ > <class> __JobShop__ > <function>__ __init__ __()

  ```python
   class JobShop(object):
      def __init__(self):
          self.job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          self.weight = np.random.uniform(0, 5, len(self.job_type))
          self.machine_num = {'process_1': 5}
          self.p_ij = {'process_1': np.random.uniform(1, 20, size=(len(self.job_type), self.machine_num['process_1']))}
          self.p_j = {'process_1': np.average(self.p_ij['process_1'], axis=1)}
          self.process_list = ['process_1', 'Sink']
          self.process_all = ['process_1']
          self.part_num = 100
          self.arrival_rate = self.machine_num['process_1'] / np.average(self.p_j['process_1'])
          self.iat = 1 / self.arrival_rate
          self.IAT = np.random.exponential(scale=self.iat, size=self.part_num)
          self.job_type_list = np.random.randint(low=0, high=10, size=self.part_num)
          # due date generating factor
          self.K = {'process_1': 1}
  
          self.env, self.model, self.source = self._modeling()
          self.done = False
  
          self.mean_weighted_tardiness = 0
  ```



+ __Parallel_Machine.py__ > <class> __JobShop__ > <function>_ __modeling__ ()

  + In order to set parts' job type and inter arrival time(IAT) random in each episode, you should type this code

  ```python
  self.arrival_rate = self.machine_num['process_1'] / np.average(self.p_j['process_1'])
  self.iat = 1 / self.arrival_rate
  self.IAT = np.random.exponential(scale=self.iat, size=self.part_num)
  self.job_type_list = np.random.randint(low=0, high=10, size=self.part_num)
  
          
  ```





### Training reinforcement learning agent



We trained agent using Double Deep Q-Network (DDQN) algorithm

Followings are the learning parameters configurations for DDQN agent



+ #### Learning Parameters Configurations (Hyper-parameters)

  + __ddqn.py__ > <class> __DDQN__ > <function>__ __init__ __()

    ```python
    self.discount_factor = 0.99	
    self.learning_rate = 1e-4
    self.epsilon = 1.0
    self.epsilon_decay = 0.9999
    self.epsilon_min = 0.01
    self.batch_size = 32
    self.train_start = 1000
    self.target_update_iter = 20
    self.memory = deque(maxlen=10000)
    ```



### Testing reinforcement learning model



We tested and compared our model with some heuristic dispatching rules(WSPT, WMDD, ATC, WCOVERT) in same parameter settings. For comparing performance, we used 100 test sets, and each test set comprises 100 randomly generated parts.

Same as training sets, weight(for tardiness) and process time for each job type and machine were set fixed, but parts' __job types__ __and __ __inter arrival time(IAT)__ __were__ __randomly generated in each training episode__.

Followings are our test settings. 

```python
    job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weight = np.random.uniform(0, 5, len(self.job_type))
    machine_num = {'process_1': 5}
    p_ij = {'process_1': np.random.uniform(1, 20, size=(len(self.job_type), self.machine_num['process_1']))}
    p_j = {'process_1': np.average(self.p_ij['process_1'], axis=1)}
    process_list = ['process_1', 'Sink']
    process_all = ['process_1']
    part_num = 100
    arrival_rate = self.machine_num['process_1'] / np.average(self.p_j['process_1'])
    iat = 1 / self.arrival_rate
    IAT = np.random.exponential(scale=self.iat, size=self.part_num)
    job_type_list = np.random.randint(low=0, high=10, size=self.part_num)
    # due date generating factor
    K = {'process_1': 1}
```



#### Test Results (comparing with heuristic dispatching rules)

We calculated average value and standard deviation value of mean weighted tardiness along 100 test sets



+ RL
  
  Average mean weighted tardiness :  2.942651995314415
  
  Std mean weighted tardiness :  1.560610048825833

+ WSPT
  
  Average mean weighted tardiness :  6.528544785630188
  
  Std mean weighted tardiness :  2.8002955608404805

+ WMDD
  
  Average mean weighted tardiness :  5.249271505149725
  
  Std mean weighted tardiness :  1.856463384230094

+ ATC
  
  Average mean weighted tardiness :  4.2719255017658435
  
  Std mean weighted tardiness :  1.7712813398930556

+ WCOVERT
  
  Average mean weighted tardiness :  7.4883050130327184
  
  Std mean weighted tardiness :  1.869074944931854

  

-----------------------------------------------
