# A machine translation example

import random  # Импортируем для работы с случайным перемешиванием данных.
import string  # Импортируем для работы с символами и пунктуацией.
import re  # Импортируем для обработки текста с регулярными выражениями.
import numpy as np  # Научные вычисления
import tensorflow as tf  # Импортируем TensorFlow для построения и обучения модели машинного перевода.
from tensorflow import keras  # Импортируем Keras для создания нейронных сетей.
from tensorflow.keras import layers  # Импортируем слои Keras для построения модели.

# Загрузка текстовых данных из файла
text_file = r"C:\Users\lenovo\Downloads\spa-eng\spa-eng\spa.txt"
with open(text_file, encoding="utf-8") as f:  # Читаем файл с кодировкой UTF-8
    lines = f.read().split("\n")[
        :-1
    ]  # Разделяем текст на строки и исключение последней пустой строки
text_pairs = []
for line in lines:
    english, spanish = line.split(
        "\t"
    )  # Разделяем строки на английскую и испанскую части
    spanish = (
        "[start] " + spanish + " [end]"
    )  # В начало и в конец фразы на испанском языке добавим токены "[start]" и "[end]" соответственно
    text_pairs.append((english, spanish))  # Добавляем пары предложений в список
# print(random.choice(text_pairs))

# Перемешаем и разделим данные на обучающую, валидационную и тестовую выборки
random.shuffle(text_pairs)  # Случайное перемешивание пар предложений
num_val_samples = int(0.15 * len(text_pairs))  # Количество примеров для валидации
num_train_samples = (
    len(text_pairs) - 2 * num_val_samples
)  # Количество примеров для обучения
train_pairs = text_pairs[:num_train_samples]  # Обучающие данные
val_pairs = text_pairs[
    num_train_samples : num_train_samples + num_val_samples
]  # Валидационные данные
test_pairs = text_pairs[num_train_samples + num_val_samples :]  # Тестовые данные

# Vectorizing the English and Spanish text pairs
# Определение символов, подлежащих удалению из текста
strip_chars = (
    string.punctuation + "¿"
)  # удаляем символ ¿, не стандартного для русского языка
strip_chars = strip_chars.replace(
    "[", ""
)  # Удаляем квадратные скобки, чтобы сохранить их в токенах
strip_chars = strip_chars.replace("]", "")


# Функция стандартизации текста,
# она должна сохранить квадратные скобки [ и ], но удалить ¿ (а также все другие символы из strings.punctuation)
def custom_standardization(input_string: tf.Tensor) -> tf.Tensor:
    lowercase = tf.strings.lower(input_string)  # Приведём текст к нижнему регистру
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# ограничим словарь 15 000 наиболее часто используемых слов в каждом языке,
vocab_size = 15000
sequence_length = 20  # а длину предложений — 20 словами

# слой для обработки строк на английском языке
source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
# слой для обработки строк на испанском языке
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
    + 1,  # Предложения на исп. языке следует генерировать с одним доп. токеном,
    # потому что во время обучения нужно сместить предложение на один шаг
    standardize=custom_standardization,
)
# Конструируем словарь для каждого языка
train_english_texts = [
    pair[0] for pair in train_pairs
]  # Разделяем английские и испанские тексты из тренировочного набора
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(
    train_english_texts
)  # Обучаем (адаптируем) векторизацию на тренировочных данных
target_vectorization.adapt(train_spanish_texts)

# Preparing datasets for the translation task
# Размер пакета данных
batch_size = 64


# Форматирование данных для обучения
def format_dataset(
    eng: tf.Tensor, spa: tf.Tensor
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return (
        {
            "english": eng,  # Входное предложение на исп. языке не включает последний токен, чтобы входные данные и цели имели одинаковую длину
            "spanish": spa[
                :, :-1
            ],  # Целевое исп. предложение на один шаг впереди. Оба имеют одинаковую длину (20 слов)
        },
        spa[:, 1:],
    )


def make_dataset(
    pairs: list[tuple[str, str]]
) -> tf.data.Dataset:  # Создаем dataset для обучения
    eng_texts, spa_texts = zip(
        *pairs
    )  # Разделяем пары на английские и испанские тексты
    eng_texts = list(eng_texts)  # создаем список
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)  # Разбиваем на батчи
    dataset = dataset.map(format_dataset, num_parallel_calls=4)  # Форматируем данные
    return (
        dataset.shuffle(2048).prefetch(16).cache()
    )  # Применение кеширования в памяти для увеличения скорости обработки


# Создаем тренировочный и проверочный наборы данных
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# for inputs, targets in train_ds.take(1):
#     print(f"inputs['english'].shape: {inputs['english'].shape}")
#     print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
#     print(f"targets.shape: {targets.shape}")


# Sequence-to-sequence learning with RNNs
# GRU-based encoder
embed_dim = 256  # Размерность эмбеддингов
latent_dim = 1024  # Размер скрытого состояния GRU

# Создаем энкодер
source = keras.Input(
    shape=(None,), dtype="int64", name="english"
)  # Вход: английский текст
# Исходное предложение на англ. языке.
# Определение имени набора входных данных позволяет нам вызвать метод fit() модели для ее обучения с входным словаре
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(
    source
)  # Эмбеддинг слоя для входа
encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(
    x
)  # Двунаправленный GRU для кодирования
# Результат кодирования исходного предложения — это последний выход двунаправленного слоя GRUНе

# GRU-based decoder and the end-to-end model
past_target = keras.Input(
    shape=(None,), dtype="int64", name="spanish"
)  # Целевое предложение на испанском
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(
    past_target
)  # Эмбеддинг слоя для входа
decoder_gru = layers.GRU(latent_dim, return_sequences=True)  # GRU слой декодера
x = decoder_gru(
    x, initial_state=encoded_source
)  # закодированное исходное предложение служит
# начальным состоянием декодера GRU
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(
    x
)  # Прогноз следующего слова
seq2seq_rnn = keras.Model([source, past_target], target_next_step)  # Полная модель:
# сопоставляет исходное предложение и целевое предложение с целевым предложением на один шаг в будущем

# Training our recurrent sequence-to-sequence model
seq2seq_rnn.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
seq2seq_rnn.fit(
    train_ds, epochs=15, validation_data=val_ds
)  # Обучение в течение 15 эпох

# Translating new sentences with our RNN encoder and decoder
# Подготовка словаря для преобразования индекса предсказываемого токена в строковый токен
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


# Функция перевода нового предложения
def decode_sequence(input_sentence: str) -> str:
    tokenized_input_sentence = source_vectorization(
        [input_sentence]
    )  # Векторизация английского текста
    decoded_sentence = "[start]"  # Отделяем начало предложения
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence]
        )  # Векторизация текущего перевода
        next_token_predictions = seq2seq_rnn.predict(
            [tokenized_input_sentence, tokenized_target_sentence]
        )
        sampled_token_index = np.argmax(
            next_token_predictions[0, i, :]
        )  # Выбор слова с наибольшей вероятностью
        sampled_token = spa_index_lookup[
            sampled_token_index
        ]  # Преобразуем индекс в слово
        decoded_sentence += " " + sampled_token  # Добавляем слово в перевод
        if (
            sampled_token == "[end]"
        ):  # условие: либо достигнута макс. длина предложения, либо получен конечный токен
            break
    return decoded_sentence


# Создаем список английских текстов из тестового набора
test_eng_texts = [pair[0] for pair in test_pairs]  # Берем первый элемент
# Выполняем перевод случайных предложений из тестового набора
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))