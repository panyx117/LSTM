#-*- coding: gbk -*-
import numpy as np
from tensorflow import keras

with open('D:/语料库/第二十四回.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 创建字符索引映射
chars = sorted(list(set(text)))#获取文本中所有不重复的字符并排序
char_to_index = {char: index for index, char in enumerate(chars)}#字符到索引的映射
index_to_char = {index: char for index, char in enumerate(chars)}#索引到字符的映射

# 将文本转换为训练样本
max_sequence_length = 100 #输入序列的长度
step = 20 #步长，控制采样间隔
sentences = []
next_chars = []
for i in range(0, len(text) - max_sequence_length, step):
    sentences.append(text[i:i + max_sequence_length])
    next_chars.append(text[i + max_sequence_length])
num_samples = len(sentences)

# 创建输入和标签矩阵
x = np.zeros((num_samples, max_sequence_length, len(chars)), dtype=np.bool_)
y = np.zeros((num_samples, len(chars)), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1


model = keras.Sequential([
    keras.layers.LSTM(256, input_shape=(max_sequence_length, len(chars))), #LSTM层
    keras.layers.BatchNormalization(), #批量规范化层（加速训练过程）
    keras.layers.Dense(len(chars), activation='softmax') #输出层
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x, y, batch_size=128, epochs=20)#batch_size:批大小    epochs:迭代次数

# 生成文本

# 选择一个随机的起始文本
start_index = np.random.randint(0, len(text) - max_sequence_length - 1)
seed_text = text[start_index:start_index + max_sequence_length]

generated_text_length = 500 #生成文本的长度
generated_text = seed_text
for _ in range(generated_text_length):
    x_pred = np.zeros((1, max_sequence_length, len(chars)))
    for t, char in enumerate(seed_text):
        x_pred[0, t, char_to_index[char]] = 1
    preds = model.predict(x_pred, verbose=0)[0]
    next_char_index = np.random.choice(range(len(chars)), p=preds)
    next_char = index_to_char[next_char_index]
    generated_text += next_char
    seed_text = seed_text[1:] + next_char

print(generated_text)
num_chars = len(chars)
total_loss = 0.0
total_chars = 0
#困惑度计算
for i in range(0, len(generated_text) - max_sequence_length, step):
    sequence = generated_text[i:i + max_sequence_length]
    x_pred = np.zeros((1, max_sequence_length, num_chars), dtype=np.bool_)
    for t, char in enumerate(sequence):
        x_pred[0, t, char_to_index[char]] = 1
    preds = model.predict(x_pred, verbose=0)[0]
    next_char_index = char_to_index[generated_text[i + max_sequence_length]]
    loss = -np.log(preds[next_char_index])
    total_loss += loss
    total_chars += 1
perplexity = np.exp(total_loss / total_chars)
print("Perplexity:", perplexity)
