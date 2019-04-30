'''
This file contains code and parameters to build a CNN model

The file also contains code for saving the model to local directory 
and evaluate the preformance in:
1) Loss/Accuracy
2) Confusion Matrix
3) AUC Score

Line 52-202 contains code for running CNN on original dataset
Line 208-550 contains code for running CNN on linear filter-augmented dataset
Line 558-906 contains code for running CNN on Guassian and white/contrast-augmented dataset

Author: Boyang Wei & Xinyu Zhang
'''

### Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(8,8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
### Base Model (with original images)
train_path = 'histopathologic-cancer-detection/base_dir/train_dir'
valid_path = 'histopathologic-cancer-detection/base_dir/val_dir'
test_path = 'histopathologic-cancer-detection/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)

# Define the batch size and steps
train_batch_size = 10
val_batch_size = 10
train_steps = np.ceil(num_train_samples / train_batch_size) 
val_steps = np.ceil(num_val_samples / val_batch_size) 

### Generators
datagen = ImageDataGenerator(rescale=1.0/255)

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)

### Building Convolutional Neural Networks (CNN)
#Parameters
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

#Build Model
model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

### Train the model
model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

filepath = "histopathologic-cancer-detection/model_1"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

#Get the history log of each step (10) of batches (9000) for training set######
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=20, verbose=1,
                   callbacks=callbacks_list)
###############################################################################
#Get accuracy and loss
model.metrics_names
model.load_weights('histopathologic-cancer-detection/model_1')

val_loss, val_acc = \
model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

# Plot accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of original dataset')
plt.legend()


plt.figure(figsize=(15,10))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy of original dataset')
plt.legend()
###############################################################################
### Prediction
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor_tissue']

### AUC
roc_auc_score(y_true, y_pred)

### Confusion Matrix
test_labels = test_gen.classes
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

###Classification Report
y_pred_binary = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)

##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

#Augmented data 1
base_tile_dir = 'histopathologic-cancer-detection/train/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[1].split('\\')[1].split('.')[0])
labels = pd.read_csv("histopathologic-cancer-detection/train_labels.csv")
df_whole = df.merge(labels, on = "id")

#Remove outliers 
#All white
whiteList = ['f6f1d771d14f7129a6c3ac2c220d90992c30c10b',
             '9071b424ec2e84deeb59b54d2450a6d0172cf701', 
             'c448cd6574108cf14514ad5bc27c0b2c97fc1a83', 
             '54df3640d17119486e5c5f98019d2a92736feabc', 
             '5f30d325d895d873d3e72a82ffc0101c45cba4a8', 
             '5a268c0241b8510465cb002c4452d63fec71028a']
#Ex:
fig = plt.figure(figsize=(5,5))
fig.suptitle('ID: ' + whiteList[0])
img = cv2.imread(base_tile_dir + whiteList[0] + '.tif')
plt.imshow(img)


#All black
blackList = ['9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
#Ex:
fig = plt.figure(figsize=(5,5))
fig.suptitle('ID: ' + blackList[0])
img = cv2.imread(base_tile_dir + blackList[0] + '.tif')
plt.imshow(img)


#Remove outliers from training set
for whiteId in whiteList:
    df_whole = df_whole[df_whole['id'] != whiteId]

for blackId in blackList:
    df_whole = df_whole[df_whole['id'] != blackId]

### Random Sampling of 10000 for both 0 and 1 cases
SAMPLE_SIZE = 10000

#Class 0
df_0 = df_whole[df_whole['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
#Class 1
df_1 = df_whole[df_whole['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# Concat the dataframes
df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# Shuffle
df_data = df_data.sample(frac=1).reset_index(drop=True)
# View the numbers in each class
df_data['label'].value_counts()


###############################################################################
# Train and Test
# stratify=y creates a balanced validation set.
y = df_data['label'] #response variable

#Split by 9(df_train)/1(df_val)
df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


### Create directory separated from the entire training set
'''
Structure:
    
- augement_1
    - train_dir
        - no_tumor_tissue
        - has_tumor_tissue
    - val_dir
        - no_tumor_tissue
        - has_tumor_tissue
'''

base_dir = 'histopathologic-cancer-detection/augement_1'
os.mkdir(base_dir)

# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# create new folders inside val_dir
no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)

# Set the ID of each image to be the index of table
df_data.set_index('id', inplace=True)

### Transfer train/test images to created folder
# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])



# Transfer the training images
for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    # source path to image
    src = os.path.join('histopathologic-cancer-detection/train/', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the validation images
for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    

    # source path to image
    src = os.path.join('histopathologic-cancer-detection/train/', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

#Check if the transfer is correct
print('Training image number with no tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/train_dir/a_no_tumor_tissue'))))
print('Training image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/train_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/val_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/val_dir/b_has_tumor_tissue'))))

### Augemented Data 1 #########################################################
aug_base_dir = 'histopathologic-cancer-detection/augement_1/'
aug_train_dir_1 = aug_base_dir + '/train_dir/a_no_tumor_tissue'
aug_train_dir_2 = aug_base_dir + '/train_dir/b_has_tumor_tissue'
aug_val_dir_1 = aug_base_dir + '/val_dir/a_no_tumor_tissue'
aug_val_dir_2 = aug_base_dir + '/val_dir/b_has_tumor_tissue'

def augment_dir(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            new_img = apply(img, 'blur_sharp')
            cv2.imwrite(os.path.join(folder,filename), new_img)

#Augment and write back all images for given directory   
augment_dir(aug_train_dir_1)
augment_dir(aug_train_dir_2)
augment_dir(aug_val_dir_1)
augment_dir(aug_val_dir_2)


###############################################################################
###Continue with modeling######################################################
### Model 2 (blur-sharp)
train_path = 'histopathologic-cancer-detection/augement_1/train_dir'
valid_path = 'histopathologic-cancer-detection/augement_1/val_dir'
test_path = 'histopathologic-cancer-detection/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)

# Define the batch size and steps
train_batch_size = 10
val_batch_size = 10
train_steps = np.ceil(num_train_samples / train_batch_size) 
val_steps = np.ceil(num_val_samples / val_batch_size) 

### Generators
datagen = ImageDataGenerator(rescale=1.0/255)

IMAGE_SIZE = 94 #Adjust the size to new one
IMAGE_CHANNELS = 3

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)

### Building Convolutional Neural Networks (CNN)
#Parameters
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

#Build Model
model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (94, 94, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

###############################################################################
### Train the model
model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

filepath = "histopathologic-cancer-detection/model_2"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

#Get the history log of each step (10) of batches (9000) for training set######
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=20, verbose=1,
                   callbacks=callbacks_list)
###############################################################################

#Get accuracy and loss
model.metrics_names
model.load_weights('histopathologic-cancer-detection/model_2')

val_loss, val_acc = \
model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

# Plot accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of augmented dataset 1')
plt.legend()


plt.figure(figsize=(15,10))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy of augmented dataset 1')
plt.legend()
###############################################################################
### Prediction
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor_tissue']

### AUC
roc_auc_score(y_true, y_pred)

### Confusion Matrix
test_labels = test_gen.classes
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

###Classification Report
y_pred_binary = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

#Augmented data 2
base_tile_dir = 'histopathologic-cancer-detection/train/'
df = pd.DataFrame({'path': glob(os.path.join(base_tile_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[1].split('\\')[1].split('.')[0])
labels = pd.read_csv("histopathologic-cancer-detection/train_labels.csv")
df_whole = df.merge(labels, on = "id")

#Remove outliers 
#All white
whiteList = ['f6f1d771d14f7129a6c3ac2c220d90992c30c10b',
             '9071b424ec2e84deeb59b54d2450a6d0172cf701', 
             'c448cd6574108cf14514ad5bc27c0b2c97fc1a83', 
             '54df3640d17119486e5c5f98019d2a92736feabc', 
             '5f30d325d895d873d3e72a82ffc0101c45cba4a8', 
             '5a268c0241b8510465cb002c4452d63fec71028a']
#Ex:
fig = plt.figure(figsize=(5,5))
fig.suptitle('ID: ' + whiteList[0])
img = cv2.imread(base_tile_dir + whiteList[0] + '.tif')
plt.imshow(img)


#All black
blackList = ['9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
#Ex:
fig = plt.figure(figsize=(5,5))
fig.suptitle('ID: ' + blackList[0])
img = cv2.imread(base_tile_dir + blackList[0] + '.tif')
plt.imshow(img)


#Remove outliers from training set
for whiteId in whiteList:
    df_whole = df_whole[df_whole['id'] != whiteId]

for blackId in blackList:
    df_whole = df_whole[df_whole['id'] != blackId]

### Random Sampling of 10000 for both 0 and 1 cases
SAMPLE_SIZE = 10000

#Class 0
df_0 = df_whole[df_whole['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
#Class 1
df_1 = df_whole[df_whole['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# Concat the dataframes
df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# Shuffle
df_data = df_data.sample(frac=1).reset_index(drop=True)
# View the numbers in each class
df_data['label'].value_counts()


###############################################################################
# Train and Test
# stratify=y creates a balanced validation set.
y = df_data['label'] #response variable

#Split by 9(df_train)/1(df_val)
df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


### Create directory separated from the entire training set
'''
Structure:
    
- augement_1
    - train_dir
        - no_tumor_tissue
        - has_tumor_tissue
    - val_dir
        - no_tumor_tissue
        - has_tumor_tissue
'''

base_dir = 'histopathologic-cancer-detection/augement_2'
os.mkdir(base_dir)

# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)

# create new folders inside train_dir
no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# create new folders inside val_dir
no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)

# Set the ID of each image to be the index of table
df_data.set_index('id', inplace=True)

### Transfer train/test images to created folder
# Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])



# Transfer the training images
for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    # source path to image
    src = os.path.join('histopathologic-cancer-detection/train/', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the validation images
for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_data.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    

    # source path to image
    src = os.path.join('histopathologic-cancer-detection/train/', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

#Check if the transfer is correct
print('Training image number with no tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/train_dir/a_no_tumor_tissue'))))
print('Training image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/train_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/val_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/augement_1/val_dir/b_has_tumor_tissue'))))

### Augemented Data 2 #########################################################
aug_base_dir = 'histopathologic-cancer-detection/augement_2/'
aug_train_dir_1 = aug_base_dir + '/train_dir/a_no_tumor_tissue'
aug_train_dir_2 = aug_base_dir + '/train_dir/b_has_tumor_tissue'
aug_val_dir_1 = aug_base_dir + '/val_dir/a_no_tumor_tissue'
aug_val_dir_2 = aug_base_dir + '/val_dir/b_has_tumor_tissue'

#apply half Gaussian filter and half bright-contrast filter to training dataset
def augment_dir(folder):
    i = 1
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            if i % 2 == 0: 
                new_img = apply(img, 'weighted_filter')
            else:
                new_img = apply(img, 'bright_contrast')
            cv2.imwrite(os.path.join(folder,filename), new_img)
            i = i + 1

#Augment and write back all images for given directory   
augment_dir(aug_train_dir_1)
augment_dir(aug_train_dir_2)
augment_dir(aug_val_dir_1)
augment_dir(aug_val_dir_2)


###############################################################################
###Continue with modeling######################################################
### Model 3 (half Gassian blur and half white/contrast filter)
train_path = 'histopathologic-cancer-detection/augement_2/train_dir'
valid_path = 'histopathologic-cancer-detection/augement_2/val_dir'
test_path = 'histopathologic-cancer-detection/test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)

# Define the batch size and steps
train_batch_size = 10
val_batch_size = 10
train_steps = np.ceil(num_train_samples / train_batch_size) 
val_steps = np.ceil(num_val_samples / val_batch_size) 

### Generators
datagen = ImageDataGenerator(rescale=1.0/255)

IMAGE_SIZE = 94 #Adjust the size to new one
IMAGE_CHANNELS = 3

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)

### Building Convolutional Neural Networks (CNN)
#Parameters
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

#Build Model
model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (94, 94, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

###############################################################################
### Train the model
model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

filepath = "histopathologic-cancer-detection/model_3"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

#Get the history log of each step (10) of batches (9000) for training set######
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=20, verbose=1,
                   callbacks=callbacks_list)
###############################################################################

#Get accuracy and loss
model.metrics_names
model.load_weights('histopathologic-cancer-detection/model_3')

val_loss, val_acc = \
model.evaluate_generator(test_gen, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

# Plot accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15,10))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of augmented dataset 2')
plt.legend()


plt.figure(figsize=(15,10))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy of augmented dataset 2')
plt.legend()
###############################################################################
### Prediction
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor_tissue']

### AUC
roc_auc_score(y_true, y_pred)

### Confusion Matrix
test_labels = test_gen.classes
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

###Classification Report
y_pred_binary = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)