import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, CSVLogger
from loguru import logger
class lstm_model:
  def __init__(self,
               num_encoder_tokens,num_decoder_tokens,
               latent_dim, encoder_input_data,
               decoder_input_data, decoder_target_data,
               batch_size, epochs
               ) -> None:
    self.num_encoder_tokens = num_encoder_tokens
    self.num_decoder_tokens = num_decoder_tokens
    self.latent_dim = latent_dim
    self.encoder_input_data = encoder_input_data
    self.decoder_input_data = decoder_input_data
    self.decoder_target_data = decoder_target_data
    self.batch_size = batch_size
    self.epochs = epochs
  def encoder(self):
    self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
    encoder = LSTM(self.latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
    self.encoder_states = [state_h,state_c]
  def decoder(self):
    self.decoder_inputs = Input(shape=(None,self.num_decoder_tokens))
    decoder = LSTM(self.latent_dim,return_sequences=True,return_state=True)
    self.decoder_outputs,_,_ = decoder(self.decoder_inputs, initial_state=self.encoder_states)
    decoder_dense = Dense(self.num_decoder_tokens,activation="softmax")
    self.decoder_outputs = decoder_dense(self.decoder_outputs)
  def model(self):
    self.model = Model([self.encoder_inputs,self.decoder_inputs], self.decoder_outputs)
    metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')]
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = metrics)
  def training(self):
    checkpoint = ModelCheckpoint(
        'trained_model.tf', 
        'val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max',
        #restore_best_weights = True,
    )
    early_stopping = EarlyStopping(monitor='loss',patience=5)
    filename='history.csv'
    history_logger=CSVLogger(filename, separator=",", append=True)
    callbacks_list = [checkpoint,early_stopping,history_logger]

    self.model.fit([self.encoder_input_data,self.decoder_input_data],self.decoder_target_data,
                   batch_size = self.batch_size,
                   callbacks=callbacks_list,
                   epochs= self.epochs, validation_split = 0.2) 