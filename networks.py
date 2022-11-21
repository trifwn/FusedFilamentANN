import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import os
import random


def create_model(modelArch,modelName,inlen):
  model = tf.keras.models.Sequential([
      Input(shape=(inlen,)),
      *[Dense(units=u,
            activation=fn,
            use_bias=t,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            ) for u,fn,t in modelArch]
  ],name=modelName)

  return model

def compile_and_fit(CASE,model,optimizer,lossfn,metrics,name, x,y,x_v=None,y_v=None,batch_size=10,lr=5e-2, verbose=0, patience=50, MAX_EPOCHS=500, record=True):

    now = datetime.now()
    dt_string = now.strftime("%H-%M-%S")

    NAME = CASE + "@"+name +"@"+dt_string

    filename = os.path.join(CASE,  NAME + '.h5')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')
    if record == True:
        tensorboard = TensorBoard(log_dir= os.path.join("logs",CASE,NAME),histogram_freq=1)
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min', save_weights_only=False)
        callbacks = [early_stopping, checkpoint]
    else:
        callbacks = [early_stopping]

    model.compile(optimizer=optimizer,loss=lossfn, metrics=metrics)

    if x_v is not None:
        history = model.fit(x,y,verbose=verbose,epochs=MAX_EPOCHS,
                            validation_data=(x_v,y_v),
                            batch_size = batch_size,
                            callbacks=callbacks) 
    else:
        filename = os.path.join(CASE,  NAME + 'k.h5')
        checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=0,
                                     save_best_only=True, mode='min', save_weights_only=False)
        history = model.fit(x,y,verbose=verbose,epochs=MAX_EPOCHS,
                            batch_size = batch_size,
                            callbacks=[tensorboard,checkpoint])
    return history