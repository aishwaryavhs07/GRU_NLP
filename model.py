import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        # ...

        self.gru_layer= tf.keras.layers.GRU(hidden_size,return_sequences=True)
        self.bidir = tf.keras.layers.Bidirectional(self.gru_layer)
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...

        M= tf.tanh(rnn_outputs)
        alpha= tf.nn.softmax(tf.tensordot(M,self.omegas,axes=1))

        r= rnn_outputs*alpha
        r=tf.reduce_sum(r,axis=1)
        output= tf.tanh(r)
       

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        tokens_mask = tf.cast(inputs!=0, tf.float32)
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...

        concwords= tf.concat([word_embed,pos_embed],axis=2)
        #only word embed
        #concwords= word_embed
        outputlayer= self.bidir(concwords,mask= tokens_mask)
        attoutput= self.attn(outputlayer)
        logits= self.decoder(inputs=attoutput)

        ### TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        #print("embed dim",embed_dim)
        self.num_classes = len(ID_TO_CLASS)
        
        self.conv2= tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2),activation='relu')
        self.pool2= tf.keras.layers.GlobalMaxPool2D()
        self.flat= tf.keras.layers.Flatten()
        self.dense1= tf.keras.layers.Dense(units=self.num_classes, activation="softmax")

        ### TODO(Students END


    def call(self, inputs, pos_inputs, training):
        
        ### TODO(Students) START
        # ...
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        concwords= tf.concat([word_embed,pos_embed],axis=2)
        concwords1= tf.expand_dims(concwords,-1)
        output= self.conv2(concwords1)
        logits = self.pool2(output)
        logits= self.flat(logits)
        logits= self.dense1(logits)
        return {'logits': logits}
        ### TODO(Students END

        

