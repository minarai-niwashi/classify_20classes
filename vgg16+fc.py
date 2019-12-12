#data preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(r"train_data_dir",target_size=(150,150),batch_size=30,class_mode="categorical")
valid_datagen = ImageDataGenerator(rescale=1.0/255)
valid_generator = train_datagen.flow_from_directory(r"valid_data_dir",target_size=(150,150),batch_size=25,class_mode="categorical")

#import VGG16
from keras.applications import VGG16

vgg16_model = VGG16(include_top=False,weights="imagenet",input_shape=(150,150,3))

#VGG16+FC
from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(vgg16_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(20,activation="softmax"))

vgg16_model.trainable = True

set_trainable = False
for layer in vgg16_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#training
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=10, verbose=1)
model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001,momentum=0.9),metrics=["acc"])
history = model.fit_generator(train_generator,steps_per_epoch=1500,epochs=200,validation_data=valid_generator,validation_steps=200,verbose=1,callbacks=[early_stopping])

model.save("model_earlystopping.h5")

#plot images
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc_values = history_dict["acc"]
val_acc_values = history_dict["val_acc"]

epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,"bo",label="Training Loss")
plt.plot(epochs,val_loss_values,"b",label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs,acc_values,"bo",label="Training Acc")
plt.plot(epochs,val_acc_values,"b",label="Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show()
