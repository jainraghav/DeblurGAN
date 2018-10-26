import numpy as np
from PIL import Image
import argparse

from deblur_gan_models import generator_model
from deblur_gan_train import load_images

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeblurGAN tester',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-p",'--datapath', type=str, default='./images/test', help='Path to testing data')
        parser.add_argument("-b",'--batch_size', type=int, default=4, help='Size of batch')
        parser.add_argument("-w",'--weights', type=str, default='generator.h5', help='Generator Weights')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def main_test(datapath,batch_size,weights):
    data = load_images(datapath, batch_size)
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights(weights)
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))

if __name__ == "__main__":
    args = Options().parse()
    main_test(args.datapath,args.batch_size,args.weights)
