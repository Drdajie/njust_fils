## 工作二：基于打分器

### 直接测试

1. cd ./t5_scorer/
2. bash run.sh

### 训练

#### 步骤一：训练打分器

1. cd ./step1_ori/
2. bash job.sh

#### 步骤二：微调T5

1. cd ./t5_scorer/
2. 将 run.sh 中注释掉的 --train 解除注释
3. bash run.sh



## 工作三：迭代利用打分器

### 直接测试

1. cd ./t5_iterator/
2. bash run.sh

### 训练

#### 步骤一：训练打分器

1. cd ./step1/
2. bash job.sh

#### 步骤二：微调T5

1. cd ./t5_iterator/
2. 将 run.sh 中 --train 的注释解除
3. bash run.sh