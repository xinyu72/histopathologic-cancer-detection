'''
This file contains code to get the original dataset and clean with two procedures:
1) Reomove outliers 
2) Balance the training and testing set

This file will create new directory and copy the images to it accordingly


Author: Boyang Wei & Xinyu Zhang
'''


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
    
- base_dir
    - train_dir
        - no_tumor_tissue
        - has_tumor_tissue
    - val_dir
        - no_tumor_tissue
        - has_tumor_tissue
'''

base_dir = 'histopathologic-cancer-detection/base_dir'
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
      str(len(os.listdir('histopathologic-cancer-detection/base_dir/train_dir/a_no_tumor_tissue'))))
print('Training image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/base_dir/train_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/base_dir/val_dir/b_has_tumor_tissue'))))
print('Test image number with tumor tissues: ' + 
      str(len(os.listdir('histopathologic-cancer-detection/base_dir/val_dir/b_has_tumor_tissue'))))