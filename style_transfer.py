import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from PIL import Image
import time
import random

import tensorflow as tf

import IPython.display

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

models = tf.keras.models
kp_image = tf.keras.preprocessing.image

# content
content_path = 'images/test.jpg'
# style
dict_of_albums = {'./covers/a-moon-shaped-pool.jpg': 'A Moon Shaped Pool',
                  './covers/amnesiac.jpg': 'Amnesiac',
                  './covers/hail-to-the-thief.jpg': 'Hail To The Thief',
                  './covers/in-rainbows.jpg': 'In Rainbows',
                  './covers/kid-a.jpg': 'Kid A',
                  './covers/ok-computer.jpg': 'OK Computer',
                  './covers/pablo-honey.jpg': 'Pablo Honey',
                  './covers/the-bends.jpg': 'The Bends',
                  './covers/the-king-of-limbs.jpg': 'The King of Limbs'}

# list_of_paths = ['./covers/a-moon-shaped-pool.jpg', './covers/amnesiac.jpg', './covers/hail-to-the-thief.jpg',
#                  './covers/in-rainbows.jpg', './covers/kid-a.jpg', './covers/ok-computer.jpg',
#                  './covers/pablo-honey.jpg', './covers/the-bends.jpg', './covers/the-king-of-limbs.jpg']
list_of_paths = list(dict_of_albums.keys())
style_path = random.choice(list_of_paths)


# чтение с диска и преобразование изображения
def load_img(path_to_img):
    max_dim = 512   # максимальная размерность
    img = Image.open(path_to_img)   # читаем с диска
    long = max(img.size)    # находим максимальную из размерностей
    scale = max_dim/long  # масшабирующий коэффициент
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.LANCZOS)   # масштабируем
    # юзаем LANCZOS

    img = kp_image.img_to_array(img)  # переводим в массив numpy

    # первая размерность - пакеты (батчи)
    img = np.expand_dims(img, axis=0)
    return img


# функция для рисования
def imshow(img, title=None):
    # для рисвоания размерность батчей не нужна, уберем ее
    out = np.squeeze(img, axis=0)
    # переводим в тип uint8
    out = out.astype('uint8')
    plt.imshow(out)   # рисуем
    if title is not None:
        plt.title(title)  # заголовок, если есть
    plt.imshow(out)

    # plt.figure(figsize=(10,10))
    #
    # content = load_img(content_path).astype('uint8')
    # style = load_img(style_path).astype('uint8')
    #
    # plt.subplot(1, 2, 1)
    # imshow(content, 'Content Image')
    #
    # plt.subplot(1, 2, 2)
    # imshow(style, 'Style Image')
    # plt.show()


def load_and_process_img(path_to_img):
    img = load_img(path_to_img) # загружаем изображение
    img = tf.keras.applications.vgg19.preprocess_input(img) # обрабатываем его для VGG
    return img


# обратный препроцессинг
def deprocess_img(processed_img):
    x = processed_img.copy()  # копия изображения
    if len(x.shape) == 4: # если 4 канала
        x = np.squeeze(x, 0)  # отрезаем первое измерение (батчи)
    # если 3, то ничего не делаем
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:   # а если не 3 и не 4, то ошибка
        raise ValueError("Invalid input to deprocessing image")

    # добавляем "среднее", которое раньше вычитали
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]   # меняем для каналов цвета (3 измерение) порядок обратно на RGB

    x = np.clip(x, 0, 255).astype('uint8')  # обрезаем до диапазона 0...255 и переводим в uint8
    return x


# название слоев из нейронной сети которые используем для признаков содержания
content_layers = ['block5_conv2']

# название слоев из нейронной сети которые используем для признаков стиля
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# загрузка модели нейронной сети
def get_model():
    # загружаем сеть VGG, обученную на данных imagenet
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # выключаем обучение весов этой сети, мы обучаем только входы в нее.
    # возвращаем выходы слоев признаков
    style_outputs = [vgg.get_layer(name).output for name in style_layers] # стиля
    content_outputs = [vgg.get_layer(name).output for name in content_layers] # содержания
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    # base_content - признаки изображения-содержания
    # target - признаки генерируемого изображения
    return tf.reduce_mean(tf.square(base_content - target))


# расчет матрицы Грамма
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])  # количество каналов = фильтров
    # многомерный массив переформатируем в двумерный, второе измерение - каналы
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]  # число признаков каждого канала
    gram = tf.matmul(a, a, transpose_a=True)  # матричное произведение
    return gram / tf.cast(n, tf.float32)  # делим на количество признаков


# ошибка стиля
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)  # считаем матрицу грамма изображения-стиля

    # height, width, channels = base_style.get_shape().as_list()
    # делим на нормирующий коэффициент / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)  # содержание
    style_image = load_and_process_img(style_path)  # стиль

    # подаем их на нейронную сеть и возвращаем признаки
    style_outputs = model(style_image)  # стиль
    content_outputs = model(content_image)  # содержание

    # собираем признаки тех слоев, что выбрали для стиля и содержания
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights   # веса ошибки стиля и содержания

    # подаем начальное изображение в нейронную сеть, возвращаем признаки для него
    model_outputs = model(init_image)
    # из них к стилю относятся первые num_style_layers выходов (величина определена ранее)
    style_output_features = model_outputs[:num_style_layers]
    # остальные выходы относятся к содержанию
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0   # ошибка стиля
    content_score = 0   # ошибка содержания

    # веса для разных слоев слагаемых ошибки стиля, равные
    weight_per_style_layer = 1.0 / float(num_style_layers)
    # перебираем все выбранные слои для стиля
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        # target_style - значения матрицы Грамма выбранных слоев признаков для изображения-стиля
        # comb_style - посчитанные выше признаки выбранных слоев обучаемого изображения
        # считаем ошибку стиля со всех выбранных слоев с весами.
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # веса для разных слоев слагаемых ошибки содержания, равные
    weight_per_content_layer = 1.0 / float(num_content_layers)
    # перебираем все выбранные слои для содержания
    for target_content, comb_content in zip(content_features, content_output_features):
        # target_content - значения выбранных слоев признаков для изображения-содержания
        # comb_content - посчитанные выше признаки выбранных слоев обучаемого изображения
        # считаем ошибку содержания со всех выбранных слоев с весами.
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    style_score *= style_weight     # умножаем на вес ошибку стиля
    content_score *= content_weight     # умножаем на вес ошибку содержания

    # общая ошибка
    loss = style_score + content_score
    return loss, style_score, content_score


# вычисление градиентов
def compute_grads(cfg):
    # принимаем некоторый объект, содержащий перечень всех наших переменных (тензоров)
    with tf.GradientTape() as tape:  # проверяем что tf.GradientTape() существует и работает
        # вычисляем ошибки, при этом будет вестись запись всех расчетов, по которой затем будет считаться градиент
        all_loss = compute_loss(**cfg)
    # считаем градиент
    total_loss = all_loss[0]  # общая ошибка
    # tape.gradient считает градиент
    return (tape.gradient(total_loss,   # тензор от которого ищем градиент
                          cfg['init_image']), # тензор по которому ищем градиент
            all_loss  # возвращаем и сами ошибки
            )


# цикл обучения
def run_style_transfer(content_path,  # путь к изображению-содержанию
                       style_path,  # путь к изображению-стилю
                       num_iterations=2000,   # число итераций обучения
                       content_weight=1e3,  # вес содержания
                       style_weight=1e4):  # вес стиля

    model = get_model()   # получаем модель нейронной сети
    for layer in model.layers:    # нам не надо обучать веса самой модели,
        layer.trainable = False   # ставим для всех слоев запрет на обучение

    # получаем признаки для изображений стиля и содержания (конкретные слои мы выбрали ранее)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    # рассчитываем матрицы Грамма для изображения-стиля
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # здесь используем изображение-содержание
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)  # переводим в Variable с автодифференцированием

    # Параметры переноса
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-1)

    # счетчик итераций
    # iter_count = 1

    # место для запоминания лучшего сгенерированного изображения и ошибки на нем
    best_loss, best_img = float('inf'), None

    # веса ошибок стиля и содержания как список
    loss_weights = (style_weight, content_weight)
    # объект (словарь) содержащий необходимые тензоры и объекты
    cfg = {
        'model': model, # модель нейронной сети
        'loss_weights': loss_weights, # веса ошибок
        'init_image': init_image, # начальное изображение
        'gram_style_features': gram_style_features, # матрицы Грамма изображения-стиля
        'content_features': content_features # признаки изображения-содержания
    }

    # для рисования
    num_rows = 2  # строки
    num_cols = 5  # столбцы
    display_interval = num_iterations/(num_rows*num_cols) # как часто рисовать
    start_time = time.time()  # запуск таймера
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68]) # "среднее" изображение
    # диапазон чисел для изображения
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []

    # Обучение
    for i in range(num_iterations):   # в цикле по количеству итераций
        grads, all_loss = compute_grads(cfg)    # вычисляем ошибки и градиент
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])    # применяем вычисленный градиент, изменяем init_image
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)    # обрезаем до нужного диапазона
        init_image.assign(clipped)
        # end_time = time.time()

    if loss < best_loss:  # если ошибки меньше предыдущей лучшей
        # обновляем лучшие
        best_loss = loss # ошибку
        best_img = deprocess_img(init_image.numpy()) # и изображение, переведя в исходный вид

    if i % display_interval == 0:   # каждые несколько итераций
        start_time = time.time()  # обновляем таймер

        plot_img = init_image.numpy()   # переводим текущее изображение из тензора Keras в numpy
        plot_img = deprocess_img(plot_img)   # возвращаем его в исходный вид
        imgs.append(plot_img)   # добавляем в массив для рисования
        IPython.display.clear_output(wait=True)   # очищаем выход текущей ячейки
        IPython.display.display_png(Image.fromarray(plot_img))   # рисуем текущее изображение
        # выводим информацию о процессе обучения
        print('Iteration: {}'.format(i))
        print('Total loss: {:.4e}, '
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))

    # По завершению обучения
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    # рисуем все сохраненные изображения
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss  # возвращаем лучшее изображение и его ошибку


# проверка работы
best, best_loss = run_style_transfer(content_path,
                                     style_path,
                                     num_iterations=300)


# посмотреть на полученное изображение
def show_results(best_img, style_path):
    plt.figure()
    plt.imshow(best_img)
    plt.title(dict_of_albums[style_path])
    plt.axis('off')
    plt.savefig("images/result.jpg")
    plt.close()


show_results(best, style_path)
