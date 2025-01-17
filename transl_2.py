# A machine translation example

import random  # Импортируем для работы с случайным перемешиванием данных.
import string  # Импортируем для работы с символами и пунктуацией.
import re  # Импортируем для обработки текста с регулярными выражениями.
import numpy as np  # Научные вычисления
import tensorflow as tf  # Импортируем TensorFlow для построения и обучения модели машинного перевода.
from tensorflow import keras  # Импортируем Keras для создания нейронных сетей.
from tensorflow.keras import layers  # Импортируем слои Keras для построения модели.


# Загрузка текстовых данных из файла
text_file = r"C:\Users\lenovo\Downloads\russian_spanish.txt"
with open(text_file, encoding="utf-8") as f:  # Читаем файл с кодировкой UTF-8
    lines = f.read().split("\n")[
        :-1
    ]  # Разделяем текст на строки и исключение последней пустой строки
text_pairs = []
for line in lines:
    russian, spanish = line.split("\t")  # Разделяем строки на русскую и испанскую части
    spanish = (
        "[start] " + spanish + " [end]"
    )  # В начало и в конец фразы на испанском языке добавим токены "[start]" и "[end]" соответственно
    text_pairs.append((russian, spanish))  # Добавляем пары предложений в список
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

# Vectorizing the russian and Spanish text pairs
# Определение символов, подлежащих удалению из текста
strip_chars = (
    string.punctuation + "¿"
)  # удаляем символ ¿, не стандартного для русского языка
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


# Функция стандартизации текста,
# она должна сохранить квадратные скобки [ и ], но удалить ¿ (а также все другие символы из strings.punctuation)
def custom_standardization(input_string: tf.Tensor) -> tf.Tensor:
    lowercase = tf.strings.lower(input_string)  # Приведём текст к нижнему регистру
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# ограничим словарь 20 000 наиболее часто используемых слов в каждом языке,
vocab_size = 20000
sequence_length = 20  # а длину предложений — 20 словами

# слой для обработки строк на русском языке
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
train_russian_texts = [
    pair[0] for pair in train_pairs
]  # Разделяем русские и испанские предложения из тренировочного набора.
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(
    train_russian_texts
)  # Обучаем (адаптируем) векторизацию на тренировочных данных
target_vectorization.adapt(train_spanish_texts)

# Preparing datasets for the translation task
# Размер пакета данных
batch_size = 64


# Форматирование данных для обучения
def format_dataset(
    rus: tf.Tensor, spa: tf.Tensor
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    rus = source_vectorization(rus)
    spa = target_vectorization(spa)
    return (
        {
            "russian": rus,
            "spanish": spa[:, :-1],
        },  # Входное предложение на исп. языке не включает последний токен, чтобы входные данные и цели имели одинаковую длину
        spa[
            :, 1:
        ],  # Целевое исп. предложение на один шаг впереди. Оба имеют одинаковую длину (20 слов)
    )


def make_dataset(
    pairs: list[tuple[str, str]]
) -> tf.data.Dataset:  # Создаем dataset для обучения
    rus_texts, spa_texts = zip(*pairs)  # Разделяем пары на русские и испанские тексты
    rus_texts = list(rus_texts)  # создаем список
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((rus_texts, spa_texts))
    dataset = dataset.batch(batch_size)  # Разбиваем на батчи
    dataset = dataset.map(format_dataset, num_parallel_calls=4)  # Форматируем данные
    return (
        dataset.shuffle(2048).prefetch(16).cache()
    )  # Применение кеширования в памяти для увеличения скорости обработки


# Создаем тренировочный и проверочный наборы данных
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# for inputs, targets in train_ds.take(1):
#     print(f"inputs['russian'].shape: {inputs['russian'].shape}")
#     print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
#     print(f"targets.shape: {targets.shape}")


# Sequence-to-sequence learning with Transformer
# The Transformer encoder
# Инициализирует слой энкодера трансформера
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # Размер входного вектора токенов
        self.dense_dim = dense_dim  # Размер внутреннего полносвязного слоя
        self.num_heads = num_heads  # Количество голов внимания
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(
                    dense_dim, activation="relu"
                ),  # Полносвязный слой с активацией ReLU
                layers.Dense(embed_dim),  # Проекция обратно в размерность эмбеддингов
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()  # Нормализация после внимания
        self.layernorm_2 = layers.LayerNormalization()  # Нормализация после проекции

    def call(
        self, inputs: tf.Tensor, mask: tf.Tensor = None
    ) -> tf.Tensor:  # Вычисления выполняются в методе call()
        if mask is not None:
            mask = mask[:, tf.newaxis, :]  # Преобразование формы маски.
            # Маска, слой Embedding сгенерирует двумерную маску,
            # но слой внимания должен быть трех- или четырехмерным, поэтому мы увеличиваем ранг
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )  # Вычисление внимания
        proj_input = self.layernorm_1(
            inputs + attention_output
        )  # Резидуальное соединение и нормализация
        proj_output = self.dense_proj(proj_input)  # Проекция через плотные слои
        return self.layernorm_2(
            proj_input + proj_output
        )  # Резидуальное соединение и нормализация

    def get_config(
        self,
    ) -> dict:  # Реализует сериализацию, чтобы дать возможность сохранить модель
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


# The Transformer decoder
# Инициализирует слой декодера трансформера
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(
                    dense_dim, activation="relu"
                ),  # Полносвязный слой с активацией ReLU
                layers.Dense(embed_dim),  # Проекция обратно в размерность эмбеддингов
            ]
        )
        self.layernorm_1 = (
            layers.LayerNormalization()
        )  # Нормализация после первого внимания
        self.layernorm_2 = (
            layers.LayerNormalization()
        )  # Нормализация после второго внимания
        self.layernorm_3 = layers.LayerNormalization()  # Нормализация после проекции
        self.supports_masking = (
            True  # Поддержка маскирования. Этот атрибут гарантирует,
        )
        # что слой будет распространять свою входную маску на свои выходные данные;
        # маскировка в Keras должна включаться явно.
        # Если передать маску слою, который не реализует метод compute_mask() и не поддерживает атрибут supports_masking,
        # данная строка вызовет ошибку

    # Возвращает конфигурацию слоя для сохранения и загрузки
    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config

    # Создает причинную маску для автогрессионного внимания.
    def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")  # Сгенерировать матрицу с формой
        # (длина_последовательности, длина_последовательности) с единицами в одной половине и с нулями в другой
        mask = tf.reshape(
            mask, (1, input_shape[1], input_shape[1])
        )  # Скопировать ее вдоль оси пакетов,
        # чтобы получить матрицу с формой (размер_пакета, длина_последовательности, длина_последовательности)
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)  # Повторение маски для каждого батча

    # Выполняет прямой проход через декодер
    def call(
        self, inputs: tf.Tensor, encoder_outputs: tf.Tensor, mask: tf.Tensor = None
    ) -> tf.Tensor:
        causal_mask = self.get_causal_attention_mask(
            inputs
        )  # Получить каузальную маску
        if (
            mask is not None
        ):  # Подготовить входную маску (описывающую точку заполнения в целевой последовательности)
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32"
            )  # Преобразование формы маски
            padding_mask = tf.minimum(padding_mask, causal_mask)  # Объединение масок
        else:
            padding_mask = mask
        # Первое внимание (авторегрессия)
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )  # Передать каузальную маску в первый слой внимания,
        # который реализует внутреннее внимание для целевой последовательности
        attention_output_1 = self.layernorm_1(
            inputs + attention_output_1
        )  # Резидуальное соединение и нормализация
        # Второе внимание (с энкодером)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )  # Передать объединенную маску во второй слой внимания,
        # который связывает исходную и целевую последовательности
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2
        )  # Резидуальное соединение и нормализация
        # Полносвязные слои и нормализация
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)  # Результат декодера


# Putting it all together: A Transformer for machine translation
# End-to-end Transformer
embed_dim = 256  # Размерность эмбеддингов
dense_dim = 2048  # Размер полносвязного слоя
num_heads = 8  # Количество голов в многоголовом внимании

# Вход для русских предложений (имя входа "russian")
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="russian")
x = layers.Embedding(vocab_size, embed_dim)(
    encoder_inputs
)  # Векторизация входных данных
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(
    x
)  # Кодирование исходной последовательности
# Вход для испанских предложений (имя входа "spanish")
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = layers.Embedding(vocab_size, embed_dim)(
    decoder_inputs
)  # Векторизация целевых данных
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(
    x, encoder_outputs
)  # Кодирование целевой последовательности
# и объединение ее с закодированной исходной последовательностью
x = layers.Dropout(0.5)(x)  # Регуляризация для предотвращения переобучения
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(
    x
)  # Предсказание слова в каждой позиции в выходе
# Создание модели
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Training the sequence-to-sequence Transformer
# Компиляция модели
transformer.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Обучение модели
transformer.fit(train_ds, epochs=15, validation_data=val_ds)

# Translating new sentences with our Transformer model
# Построение словаря для испанского языка
spa_vocab = target_vectorization.get_vocabulary()  # Получение словаря из векторизатора
spa_index_lookup = dict(
    zip(range(len(spa_vocab)), spa_vocab)
)  # Создание словаря {индекс: слово}
max_decoded_sentence_length = 20  # Максимальная длина предложения при переводе


# Функция для перевода нового предложения
def decode_sequence(input_sentence: str) -> str:
    tokenized_input_sentence = source_vectorization(
        [input_sentence]
    )  # Векторизация русского текста
    decoded_sentence = "[start]"  # Инициализация начала перевода
    for i in range(max_decoded_sentence_length):  # Ограничение длины перевода
        tokenized_target_sentence = target_vectorization([decoded_sentence])[
            :, :-1
        ]  # Векторизация перевода
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence]
        )  # Предсказание следующего слова
        sampled_token_index = np.argmax(
            predictions[0, i, :]
        )  # Индекс слова с максимальной вероятностью
        sampled_token = spa_index_lookup[
            sampled_token_index
        ]  # Получение слова из словаря
        decoded_sentence += " " + sampled_token  # Добавление слова к переводу
        if sampled_token == "[rus]":  # Прерывание перевода, если достигнут токен конца
            break
    return decoded_sentence


# Список русских предложений для тестирования
test_rus_texts = [
    pair[0] for pair in test_pairs
]  # Извлечение русских предложений из тестовых пар
for _ in range(20):  # Переводим 20 случайных предложений
    input_sentence = random.choice(
        test_rus_texts
    )  # Случайный выбор предложения из тестового набора
    print("-")
    print(input_sentence)  # Печатаем русское предложение
    print(decode_sequence(input_sentence))  # Переводим и выводим испанское предложение

# Результат обучения:
# Epoch 15/15
# accuracy: 0.9084 - loss: 0.5652 - val_accuracy: 0.8621 - val_loss: 0.9957


# Несколько примеров перевода:
# Нарисуй кружочек.
# [start] [UNK] un pequeño [UNK] [end]

# Стакан был наполовину полон.
# [start] el vaso estaba lleno [end]

# Эта юбка длинная.
# [start] esta falda es larga [end]

# Сделаю, если будет время.
# [start] el tiempo si es de la mañana [end]

# В Испании подарки детям приносят три короля-мага.
# [start] en españa se [UNK] los niños de españa [end]

# Том дал мне твой номер.
# [start] tom me dio tu número [end]

# У моего отца есть ресторан.
# [start] mi padre tiene un restaurante [end]

# Я обнаружил комнату пустой.
# [start] encontré una habitación vacía [end]

# Без тебя жизнь уже никогда не будет такой, как прежде, любовь моя.
# [start] el amor no es como el mundo antes que el mundo sin ti mi vida [end]

# Ты не против посторожить минутку мой чемодан?
# [start] no estás en contra mi maleta momento [end]

# У пары семеро детей.
# [start] los niños tienen siete años [end]

# Мысли выражаются словами.
# [start] las palabras [UNK] en palabras [end]

# Том - талантливый актёр.
# [start] tom es un actor talentoso [end]
# -
# Том жил рядом с Марией.
# [start] tom vivía con maría [end]
