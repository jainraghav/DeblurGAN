import numpy as np
from PIL import Image
import argparse

from deblur_gan_models import generator_model
from deblur_gan_train import preprocess
from deblur_gan_test import deprocess_image

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeblurGAN demo',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-i",'--imgpath', type=str, help='Input image')
        parser.add_argument("-w",'--weights', type=str, default='generator.h5', help='Generator Weights')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def demo(image_path,weights):
    img_test = Image.open(image_path)
    data = {'A_paths': [image_path],'A': np.array([preprocess(img_test)])}
    x_test = data['A']
    g = generator_model()
    g.load_weights(weights)
    generated_images = g.predict(x=x_test)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)

    for i in range(generated_images.shape[0]):
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('deblur'+image_path)

if __name__ == "__main__":
    args = Options().parse()
    demo(args.imgpath,args.weights)
