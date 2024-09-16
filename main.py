from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow import keras
import numpy as np

text = '''
  «Но повесть о Ромео и Джульетте останется печальнейшей на свете...»
  Самой печальной и, пожалуй, самой известной историей любви в литературе.
  Написанная в конце XVI века, трагедия Шекспира до сих пор не сходит с театральной сцены,
  а в ХХ веке еще и десятки раз была экранизирована. В ее основу положена старинная легенда
  о трагической судьбе юноши и девушки из враждующих семейств. Под пером великого драматурга
  этот «бродячий» сюжет обрел величие и бессмертие, став гимном свободы человеческой личности,
  порывающей с миром косных средневековых законов, гимном лучезарной, истинной любви.
  В настоящем издании трагедия «Ромео и Джульетта» представлена в классическом переводе Бориса Пастернака.
'''

num_characters = 34
tokenizer = Tokenizer(num_words=num_characters, char_level=True)
tokenizer.fit_on_texts([text])

inp_chars = 6
data = tokenizer.texts_to_matrix(text)
n = data.shape[0] - inp_chars

x = np.array([data[i: i + inp_chars, :] for i in range(n)])
y = data[inp_chars:]

model = keras.Sequential([
    Input((inp_chars, num_characters)),
    SimpleRNN(128, activation='tanh'),
    Dense(num_characters, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
hist = model.fit(x, y, batch_size=32, epochs=100)

def buildPhrase(inp_str, str_len=50):
  for i in range(str_len):
    x = []
    for j in range(i, i + inp_chars):
      x.append(tokenizer.texts_to_matrix(inp_str[j]))

    x = np.array(x)
    inp = x.reshape(1, inp_chars, num_characters)

    pred = model.predict(inp)
    d = tokenizer.index_word[pred.argmax(axis=1)[0]]

    inp_str += d

  return inp_str

res = buildPhrase('утренн')
print(res)