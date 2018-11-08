## Chinese Intent Classify 2018-11

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词，统一替换地区、时间

等特殊词，word_replace() 替换同音、同义词，打乱后划分训练、验证、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

embed() 通过 word_inds 与 word_vecs 建立词索引到词向量的映射

label2ind() 建立标签索引的字典，sorted() 保持一致性

align() 将序列填充为相同长度，标签不使用 to_categorical() 编码

#### 4.build

tensorize() 将 numpy array 转换为 torch LongTensor

get_loader() 打乱并划分 batch，通过 dnn、cnn、rnn 构建分类模型

#### 5.classify

predict() 去除停用词，统一替换地区、时间等特殊词，输出所有类别的概率