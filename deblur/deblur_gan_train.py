import os
import numpy as np
import datetime
import argparse
from PIL import Image
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam

from deblur_gan_models import generator_model, discriminator_model, DeblurGAN

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeblurGAN trainer',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-p",'--datapath', type=str, default='./images/train', help='Path to training data')
        parser.add_argument("-b",'--batch_size', type=int, default=16, help='Size of batch')
        parser.add_argument("-e",'--epochs',type=int, default=4, help='Number of epochs for training')
        parser.add_argument("-n",'--nimages', type=int, default=512, help='Number of images to load for training')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def is_an_image_file(filename):
    extentions = ['.png', '.jpg', '.jpeg']
    for ext in extentions:
        if ext in filename:
            return True
    return False

def preprocess(input_img):
    a_img = input_img.resize((256,256))
    img = np.array(a_img)
    img = (img - 127.5) / 127.5
    return img

def load_images(path, n_images):
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    A_files,B_files = os.listdir(A_paths),os.listdir(B_paths)
    all_A_paths = [os.path.join(A_paths, f) for f in A_files if is_an_image_file(f)]
    all_B_paths = [os.path.join(B_paths, f) for f in B_files if is_an_image_file(f)]
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A = Image.open(path_A)
        img_B = Image.open(path_B)
        images_A.append(preprocess(img_A))
        images_B.append(preprocess(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {'A': np.array(images_A),'A_paths': np.array(images_A_paths),'B': np.array(images_B),'B_paths': np.array(images_B_paths)}

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join('weights/', '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

def main_train(datapath,n_images,batch_size,epochs):
    data = load_images(datapath, n_images)
    y_train, x_train = data['B'], data['A']
    g = generator_model()
    d = discriminator_model()
    deblurgan = DeblurGAN(g, d)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    deblurgan_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    deblurgan.compile(optimizer=deblurgan_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    for epoch in range(epochs):
        print('epoch: {}/{}'.format(epoch, epochs))
        print('batches: {}'.format(x_train.shape[0] / batch_size))

        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        deblurgan_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(5):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
            print('batch {} d_loss : {}'.format(index+1, np.mean(d_losses)))

            d.trainable = False

            deblurgan_loss = deblurgan.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            deblurgan_losses.append(deblurgan_loss)
            print('batch {} deblurgan_loss : {}'.format(index+1, deblurgan_loss))

            d.trainable = True

        with open('log.txt', 'a') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(deblurgan_losses)))

        save_all_weights(d, g, epoch, int(np.mean(deblurgan_losses)))

if __name__ == '__main__':
    args = Options().parse()
    main_train(args.datapath, args.nimages, args.batch_size, args.epochs)
