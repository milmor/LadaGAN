"""LadaGAN model for Tensorflow.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import tensorflow as tf
from tensorflow.keras import layers


def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = tf.reshape(x, (-1, H, W, C))
    x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
    B, H, W, C = x.shape
    
    return x, H, W, C


class SMLayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, initializer='orthogonal'):
        super(SMLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.initializer = initializer
        
    def build(self, inputs):
        input_shape, _ = inputs
        self.h = layers.Dense(input_shape[2], use_bias=True, 
                    kernel_initializer=self.initializer
        )
        self.gamma = layers.Dense(input_shape[2], use_bias=True, 
            kernel_initializer=self.initializer, 
        )
        self.beta = layers.Dense(input_shape[2], use_bias=True, 
                        kernel_initializer=self.initializer
        )
        self.ln = layers.LayerNormalization(
            epsilon=self.epsilon, center=False, scale=False
        )

    def call(self, inputs):
        x, z = inputs
        x = self.ln(x)
        h = self.h(z)
        h = tf.nn.relu(h)
        
        scale = self.gamma(h)
        shift = self.beta(h)
        x *= tf.expand_dims(scale, 1)
        x += tf.expand_dims(shift, 1)
        return x


class AdditiveAttention(layers.Layer):
    def __init__(self, model_dim, n_heads, initializer='orthogonal'):
        super(AdditiveAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wk = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wv = layers.Dense(model_dim, kernel_initializer=initializer)
        
        self.q_attn = layers.Dense(n_heads, kernel_initializer=initializer)
        dim_head = model_dim // n_heads

        self.to_out = layers.Dense(model_dim, kernel_initializer=initializer)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        B = tf.shape(q)[0]
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  
        attn = tf.transpose(self.q_attn(q), [0, 2, 1]) / self.depth ** 0.5
        attn = tf.nn.softmax(attn, axis=-1)  
   
        q = self.split_into_heads(q, B)  
        k = self.split_into_heads(k, B)  
        v = self.split_into_heads(v, B)

        # calculate global vector
        global_q = tf.einsum('b h n, b h n d -> b h d', attn, q) 
        global_q = tf.expand_dims(global_q, 2)
       
        p = global_q * k 
        r = p * v

        r = tf.transpose(r, perm=[0, 2, 1, 3]) 
        original_size_attention = tf.reshape(r, (B, -1, self.model_dim)) 

        output = self.to_out(original_size_attention) 
        return output, attn


class SMLadaformer(layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(SMLadaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu', 
                         kernel_initializer=initializer), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        self.norm1 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.norm2 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training):
        inputs, z = x
        x_norm1 = self.norm1([inputs, z])
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output, training=training) 
        
        x_norm2 = self.norm2([attn_output, z])
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output, training=training), attn_maps 
    
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, n_patches, model_dim, initializer='orthogonal'):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = layers.Embedding(
            input_dim=n_patches, output_dim=model_dim, 
            embeddings_initializer=initializer
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)
    

class Generator(tf.keras.models.Model):
    def __init__(self, img_size=32, model_dim=[1024, 256, 64], heads=[2, 2, 2], 
                 mlp_dim=[2048, 1024, 512], initializer='orthogonal', dec_dim=False):
        super(Generator, self).__init__()
        self.init = tf.keras.Sequential([
            layers.Dense(8 * 8 * model_dim[0], use_bias=False, 
                                kernel_initializer=initializer),
            layers.Reshape((8 * 8, model_dim[0]))
        ])     
    
        self.pos_emb_8 = PositionalEmbedding(64, model_dim[0], 
                                initializer=initializer)
        self.block_8 = SMLadaformer(model_dim[0], heads[0], 
                                mlp_dim[0], initializer=initializer)
        self.conv_8 = layers.Conv2D(model_dim[1], 3, padding='same', 
                                kernel_initializer=initializer)

        self.pos_emb_16 = PositionalEmbedding(256, model_dim[1], 
                                initializer=initializer)
        self.block_16 = SMLadaformer(model_dim[1], heads[1], 
                                mlp_dim[1], initializer=initializer)
        self.conv_16 = layers.Conv2D(model_dim[2], 3, padding='same', 
                                kernel_initializer=initializer)

        self.pos_emb_32 = PositionalEmbedding(1024, model_dim[2], 
                                initializer=initializer)
        self.block_32 = SMLadaformer(model_dim[2], heads[2], 
                                mlp_dim[2], initializer=initializer)

        self.dec_dim = dec_dim
        if self.dec_dim:
            self.dec = tf.keras.Sequential()
            for _ in self.dec_dim:
                self.dec.add(layers.UpSampling2D(2, interpolation='nearest'))
                self.dec.add(layers.Conv2D(_, 3, padding='same', 
                                    kernel_initializer=initializer)),
                self.dec.add(layers.BatchNormalization())
                self.dec.add(layers.LeakyReLU(0.2))
        else:
            self.patch_size = img_size // 32
        self.ch_conv = layers.Conv2D(3, 3, padding='same', 
                                kernel_initializer=initializer)

    def call(self, z):
        B = z.shape[0]
   
        x = self.init(z)
        x = self.pos_emb_8(x)
        x, attn_8 = self.block_8([x, z])

        x, H, W, C = pixel_upsample(x, 8, 8)
        x = self.conv_8(x)
        x = tf.reshape(x, (B, H * W, -1))

        x = self.pos_emb_16(x)
        x, attn_16 = self.block_16([x, z])

        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv_16(x)
        x = tf.reshape(x, (B, H * W, -1))
        x = self.pos_emb_32(x)
        x, attn_32 = self.block_32([x, z])

        x = tf.reshape(x, [B, 32, 32, -1])
        if  self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = tf.nn.depth_to_space(x, self.patch_size, data_format='NHWC')
        return [self.ch_conv(x), [attn_8, attn_16, attn_32]]

    
class downBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, strides=2, 
                 initializer='glorot_uniform'):
        super(downBlock, self).__init__()
        self.main = tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=kernel_size, 
                padding='same', kernel_initializer=initializer,
                strides=strides, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters, kernel_size=3, 
                padding='same', kernel_initializer=initializer,
                strides=1, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.direct = tf.keras.Sequential([
           layers.AveragePooling2D(pool_size=(strides, strides)),
            layers.Conv2D(filters, kernel_size=1, 
                padding='same', kernel_initializer=initializer,
                strides=1, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

    def call(self, x):
        return (self.main(x) + self.direct(x)) / 2


class Ladaformer(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(Ladaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim, activation='gelu', kernel_initializer=initializer
            ), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=eps)
        self.norm2 = layers.LayerNormalization(epsilon=eps)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training):
        x_norm1 = self.norm1(inputs)
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output, training=training) 
        
        x_norm2 = self.norm2(attn_output)
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output, training=training) + attn_output, attn_maps

    
class Discriminator(tf.keras.models.Model):
    def __init__(self, img_size=32, enc_dim=[64, 128, 256], out_dim=[512, 1024], mlp_dim=512, 
                 heads=2, initializer='orthogonal'):
        super(Discriminator, self).__init__()
        if img_size == 32:
            assert len(enc_dim) == 2, "Incorrect length of enc_dim for img_size 32"
        elif img_size == 64:
            assert len(enc_dim) == 3, "Incorrect length of enc_dim for img_size 64"
        elif img_size == 128:
            assert len(enc_dim) == 4, "Incorrect length of enc_dim for img_size 128"
        elif img_size == 256:
            assert len(enc_dim) == 5, "Incorrect length of enc_dim for img_size 256"
        else:
            raise ValueError(f"img_size = {img_size} not supported")
            
        self.enc_dim = enc_dim
        self.inp_conv = tf.keras.Sequential([
            layers.Conv2D(enc_dim[0], kernel_size=3, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='same'),
            layers.LeakyReLU(0.2),
        ])    
        self.encoder = [downBlock(
            i, kernel_size=3, strides=2, initializer=initializer
        ) for i in enc_dim[1:]]

        self.pos_emb_8 = PositionalEmbedding(256, enc_dim[-1], 
                            initializer=initializer)
        self.block_8 = Ladaformer(enc_dim[-1], heads, 
                            mlp_dim, initializer=initializer)
        
        self.conv_4 = layers.Conv2D(out_dim[0], 3, padding='same', 
                                    kernel_initializer=initializer)
        self.down_4 = tf.keras.Sequential([
            layers.Conv2D(out_dim[1], kernel_size=1, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid')
        ])
        '''Logits'''
        self.logits = tf.keras.Sequential([
            layers.Flatten(),
            layers.Activation('linear', dtype='float32')    
        ])
    
    def call(self, img):
        x = self.inp_conv(img)  
        for i in range(len(self.enc_dim[1:])):
            x = self.encoder[i](x)

        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H * W, C))
        x = self.pos_emb_8(x)
        x, maps_16 = self.block_8(x)

        x = tf.reshape(x, (B, H, W, C))
        x = tf.nn.space_to_depth(x, 2, data_format='NHWC') 
        x = self.conv_4(x)

        x = self.down_4(x)
        return [self.logits(x)]