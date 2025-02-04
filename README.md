# Task-9_Sholle-11
## Машинный перевод с использованием рекуррентных нейронных сетей, файл my_transl.py

Этот код реализует перевод текста с использованием архитектуры Sequence-to-Sequence (Seq2Seq) на основе рекуррентных нейронных сетей (RNN) с блоками GRU (Gated Recurrent Unit). 
Эпаты реализации:
1.Подготовка данных.
2.Векторизация текста.
3.Построение модели с энкодером и декодером.
4.Обучение модели.
5.Тестирование системы на новых предложениях.

 1. Подготовка данных

*Загрузим файл с данными: нужно ввести свой путь к файлу spa.txt (папка есть в списке файлов, её нужно распаковать из архива).*
Данные представляют собой пары предложений на английском и испанском языках, разделенные табуляцией.
Файл читается построчно, после чего испанские предложения дополняются токенами [start] (начало предложения) и [end] (конец предложения).
Затем каждая строка преобразуется в пару из английского и испанского предложений.
Потом данные перемешиваются случайным образом и разделяются на:
Тренировочный набор (70% данных).
Проверочный набор (15% данных).
Тестовый набор (15% данных).

 2. Векторизация текста

Нужно преобразовать текст в числовые последовательности, подходящие для обработки нейронными сетями.
Функция стандартализации преобразует текст к нижнему регистру и удаляет знаки пунктуации, за исключением токенов [start] и [end].
Потом идёт создание объектов TextVectorization:
Для английских предложений используется стандартная векторизация.
Для испанских предложений добавляется кастомная стандартизация.
Векторизация настраивается на тренировочных данных
Потом данные разделяются на входные последовательности (исходный язык) и целевые последовательности (язык перевода).
Используется API tf.data для создания оптимизированного набора данных.

 3. Построение модели Seq2Seq

Энкодер преобразует входное предложение в скрытое представление с использованием двунаправленного GRU
Декодер генерирует выходное предложение на основе скрытого состояния энкодера
Потом объединяем всё, модель принимает два входа: последовательности на английском и уже переведенные слова на испанском

 4. Обучение модели

Модель обучается с использованием sparse_categorical_crossentropy и оптимизатора rmsprop

 5. Перевод новых предложений

Функция перевода принимает английское предложение, декодирует его по одному слову за шаг и возвращает результат.
С помощью метода random выбираются случайные тестовые предложения, и их перевод выводится в терминале.


# Task-9_Sholle-11_2
## Модель Transformer для машинного перевода, файл transl_2.py

Этот код реализует перевод текста с использованием архитектуры Transformer для выполнения задачи Sequence-to-Sequence (Seq2Seq). Transformer построен на механизмах внимания и позволяет эффективно работать с длинными последовательностями. Процесс состоит из следующих этапов:

1. Подготовка данных

**Установим все библиотеки через команду pip install -r requirements.txt в терминале**

*Загрузим файл с данными: нужно ввести свой путь к файлу russian_spanish.txt (файл есть в списке, его нужно скачать).*
Данные представляют собой пары предложений на русском и испанском языках, разделенные табуляцией.
Файл читается построчно, после чего испанские предложения дополняются токенами [start] (начало предложения) и [end] (конец предложения).
Затем каждая строка преобразуется в пару из русского и испанского предложений.
Потом данные перемешиваются случайным образом и разделяются на:
Тренировочный набор (70% данных).
Проверочный набор (15% данных).
Тестовый набор (15% данных).

2. Разделение данных

Данные случайным образом перемешиваются и разделяются на три набора:
Обучающая выборка: 70% данных.
Проверочная выборка: 15% данных.
Тестовая выборка: 15% данных.

3. Векторизация текста

Текст преобразуется в числовые последовательности с использованием слоёв TextVectorization. Это позволяет преобразовать слова в индексы, которые могут быть обработаны моделью.
Кастомная стандартизация текста
Удаляются символы пунктуации (кроме специальных токенов [start] и [end]), и текст переводится в нижний регистр.
Настройка векторизации
Для русского языка используется стандартная векторизация.
Для испанского языка добавляется кастомная стандартизация.

4. Создание наборов данных

С помощью tf.data создаются оптимизированные наборы данных. Данные разбиваются на батчи, форматируются и кешируются для ускорения обработки.

5. Построение модели Transformer

Энкодер преобразует входное предложение (на русском языке) в скрытое представление с использованием многоголового внимания и плотных слоёв.
Декодер использует скрытое представление из энкодера и текущие переведённые слова для предсказания следующего слова.
Полная модель
Объединяем энкодер и декодер для создания модели Seq2Seq.

6. Обучение

Модель обучается на тренировочных данных с использованием функции потерь sparse_categorical_crossentropy и оптимизатора rmsprop.

7. Перевод новых предложений

Функция перевода использует энкодер для кодирования входного предложения и декодер для автогрессионного предсказания перевода.
