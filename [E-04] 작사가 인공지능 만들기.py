#!/usr/bin/env python
# coding: utf-8

# # Step 1. 데이터 다운로드

# # Step 2. 데이터 읽어오기
# 

# In[1]:


# 라이브러리 불러오기
import glob
import os, re  #re가 빠지면 안됨!
import numpy as np
import tensorflow as tf

# 파일 읽기
txt_file_path = os.getenv('HOME')+'/aiffel/lyricist/data/lyrics/*'
txt_list = glob.glob(txt_file_path)
raw_corpus = []

for txt_file in txt_list:
    with open(txt_file, "r") as f:
        raw = f.read().splitlines()
        raw_corpus.extend(raw)

#파일 안에 있는 단어 개수 확인 및 2줄 불러와서 확인하기
print("데이터 크기:", len(raw_corpus))
print("Examples:\n", raw_corpus[:2])


# # Step 3. 데이터 정제

# In[5]:


for idx, sentence in enumerate(raw_corpus):
    if len(sentence)==0: continue # 문장 길이가 0 (공백)이면 건너뛰기
    if sentence[-1]=="]": continue #[hook]과 같은 형식을 나타내는 부분은 건너뛰기

    if idx > 2: break # 문장 확인
    print(sentence)


# In[6]:


# preprocess_sentence() 토큰화
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip() # 모두 소문자화, 공백 지우기
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence) #특수기호 양쪽에 공백 추가
    sentence = re.sub(r'[" "]+', " ", sentence) # 공백 한칸으로 수정 및 통일
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence) # 지정 기호 외에 공백으로 수정
    sentence = sentence.strip() # 양쪽 공백 지우기
    sentence = '<start> ' + sentence + ' <end>' # <start>, <end> 넣기
    return sentence


# In[7]:


# 정제된 문장들 모으기
corpus = []

for sentence in raw_corpus:
    if len(sentence) == 0: continue
    if sentence[-1] == "]": continue #[hook] 빼기
    preprocessed_sentence = preprocess_sentence(sentence)
    corpus.append(preprocessed_sentence)

# 정제된 문장 10개 확인
corpus[:10]


# # Step 4. 평가 데이터셋 분리

# In[8]:


# 토큰화: 텐서 플로우 tokenizer, pad_sequences 사용
# 12000개의 단어 사용
def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=12000, 
        filters=' ',
        oov_token="<unk>" # 12000개에 속하지 않으면 unk로
    )
    
    tokenizer.fit_on_texts(corpus)
    tensor = tokenizer.texts_to_sequences(corpus) 
    total_data_text = list(tensor)
    num_tokens = [len(tokens) for tokens in total_data_text]
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    maxlen = int(max_tokens)
    
    # 입력 데이터의 시퀀스 길이를 맞춰준다.
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, 
                                                           padding='post',
                                                          maxlen=maxlen)  
   # tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  
    
    print(tensor,tokenizer)
    return tensor, tokenizer

tensor, tokenizer = tokenize(corpus)


# In[9]:


# tensor에서 마지막 토큰을 잘라 문장을 생성
src_input = tensor[:, :-1]
tgt_input=tensor[:, 1:]

# 훈련 셋과 검증 셋을 나누기
from sklearn.model_selection import train_test_split
enc_train, enc_val, dec_train, dec_val = train_test_split(src_input, 
                                                          tgt_input,
                                                          test_size=0.2, # 20%의 테스트 데이터
                                                          shuffle=True, 
                                                          random_state=68)
print('Source Train: ', enc_train.shape)
print('Target Train: ', dec_train.shape)


# # Step 5. 인공지능 만들기

# In[10]:


# LYRICS GENERATOR 클래스를 통해서 모델 만들기
# 1개의 embedding, 2개의 LSTM layers, 1개의 dense layer
class LyricsGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)
        
    def call(self, x):
        out = self.embedding(x)
        out = self.rnn_1(out)
        out = self.rnn_2(out)
        out = self.linear(out)
        return out
    
embedding_size = 512  # embedding 사이즈를 높여서 모델의 성능 높이기
hidden_size = 2048
model = LyricsGenerator(tokenizer.num_words + 1, embedding_size , hidden_size)


# In[11]:


# 모델을 10번으로 나눠서 학습할 수 있도록 에폭시를 10으로 설정
model = LyricsGenerator(tokenizer.num_words + 1, embedding_size , hidden_size)
history = []
epochs = 10

optimizer = tf.keras.optimizers.Adam()

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)

model.compile(loss=loss, optimizer=optimizer)
#model.fit(dataset, epochs=10)


# In[12]:


# 데이터셋 객체 생성 및 전처리
BUFFER_SIZE = len(src_input)
BATCH_SIZE = 256
steps_per_epoch = len(src_input) // BATCH_SIZE

VOCAB_SIZE = tokenizer.num_words + 1 # "+1" = 0:<pad>를 포함   

dataset = tf.data.Dataset.from_tensor_slices((src_input, tgt_input)) #메소드
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
for src_sample, tgt_sample in dataset.take(1): break

# 한 배치만 불러온 데이터를 모델에 넣가
model(src_sample)


# In[13]:


model.summary()


# In[14]:


history = model.fit(enc_train, 
          dec_train, 
          epochs=epochs,
          batch_size=256,
          validation_data=(enc_val, dec_val),
          verbose=1)


# In[15]:


# 모델을 평가하기 위해 generate_text함수를 통해 작문 
def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    test_input = tokenizer.texts_to_sequences([init_sentence]) # 텐서로 변환
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    while True:
        # 입력받은 문장 텐서에 입력
        predict = model(test_tensor) 
        # 예측된 값중 가장 높은 것으로 인덱스 출력
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1] 
        # 출력된 인덱스를 문장뒤에 붙이기
        test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)
        # end 또는 max 길이에 넘는다면 문장 생성 종료
        if predict_word.numpy()[0] == end_token: break
        if test_tensor.shape[1] >= max_len: break

    generated = ""
    for word_index in test_tensor[0].numpy(): #토크나이저를 이용하여 인덱스를 단어로 호환
        generated += tokenizer.index_word[word_index] + " "

    return generated


# In[17]:


# 시범 문장 (입력문장): i love
generate_text(model, tokenizer, init_sentence="<start> i love", max_len=100)


# In[ ]:


print(dataset)


# In[ ]:




