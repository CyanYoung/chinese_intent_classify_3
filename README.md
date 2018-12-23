## Chinese Intent Classify 2018-11

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词，统一替换地区、时间

等特殊词，word_replace() 替换同音、同义词，打乱后划分训练、验证、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

Dictionary() 建立词索引的字典，filter_extremes() 过滤低频词

tran_dict() 为填充词、未录词保留索引 0、1，label2ind() 建立标签索引的字典

sent2ind() 将每句转换为词索引，并将序列填充为相同长度

#### 4.build

tensorize() 将 array 转换为 LongTensor，get_loader() 打乱并划分 batch

通过 dnn、cnn、rnn 构建分类模型，dev_loss 降低则保存模型

trap_count > max_count 则 learn_rate / 10、小于 min_rate 则早停止

#### 5.classify

predict() 去除停用词，统一替换地区、时间等特殊词，输出所有类别的概率