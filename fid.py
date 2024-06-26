import numpy as np
from scipy import linalg
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    
def get_activations(dataset, inception, batch_size=20):
    n_batches = tf.data.experimental.cardinality(dataset)
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048), 'float32')

    for i, batch in enumerate(tqdm(dataset)):
        start = i * batch_size
        end = start + batch_size
        pred = inception(batch)
        pred_arr[start:end] = pred # pred.reshape(batch_size, -1)
        
    return pred_arr

def calculate_activation_statistics(images, model, batch_size=20):
    act = get_activations(images, model, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(real, gen, model, batch_size=20):
    if isinstance(real, list):
        m1, s1 = real
    else:   
        m1, s1 = calculate_activation_statistics(real, model, batch_size)
        
    if isinstance(gen, list):
        m2, s2 = gen
    else:  
        m2, s2 = calculate_activation_statistics(gen, model, batch_size)
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def test_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    return img

def create_fid_ds(img_dir, batch_size, img_size, n_images, seed=42):
    img_paths = tf.data.Dataset.list_files(str(img_dir), seed=seed).take(n_images)
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)
    ds = img_paths.map(lambda img: test_convert(img, img_size), 
                       num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True, 
            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    ds_size = tf.data.experimental.cardinality(ds)
    print(f'FID dataset size: {BUFFER_SIZE} FID batches: {ds_size}')  
    return ds

class Inception(tf.keras.models.Model):
    ''' Calculates the activations
    -- inp (inception_v3.preprocess_input): A floating point numpy.array or a tf.Tensor,
            3D or 4D with 3 color channels, with values in the range [0, 255].
    '''
    def __init__(self):
        super(Inception, self).__init__()
        self.res = preprocessing.Resizing(299, 299)
        self.inception = InceptionV3(include_top=False,  pooling='avg')
        self.inception.trainable = False

    @tf.function
    def call(self, inp):
        x = self.res(inp)
        x = inception_v3.preprocess_input(x)       
        x = self.inception(x)
        return x
