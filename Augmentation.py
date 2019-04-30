'''
This file contains core functions to augment the orginal data

There are three types of aumgmentation functions:
1) Blur and Sharp (Linear Kernel)
2) Gaussian Blur
3) Random whitenning and Contrast

Function Apply will apply the function name to an input image

Main function here help visualize an example from the original training set


Author: Boyang Wei & Xinyu Zhang
'''

#Blur the surrounding 5px and sharp the inner 86px
def blur_sharp(color):
    #New imgae matrix placeholder
    matrix = np.zeros((96,96), dtype=np.uint8)
    R_new = color
    
    #Gaussian Blur Kernel on 96-86-96px
    for i in range(1,5):
        for j in range(1,95):
            matrix[i,j] = np.uint8((np.int(R_new[i-1,j-1])/16 + np.int(R_new[i-1,j])/8 + np.int(R_new[i-1,j+1])/16 + 
                           np.int(R_new[i,j-1])/8 + np.int(R_new[i,j])/4 + np.int(R_new[i,j+1])/8 +
                           np.int(R_new[i+1,j-1])/16 + np.int(R_new[i+1,j])/8 + np.int(R_new[i+1,j+1])/16))
    for i in range(91,95):
        for j in range(1,95):
            matrix[i,j] = np.uint8((np.int(R_new[i-1,j-1])/16 + np.int(R_new[i-1,j])/8 + np.int(R_new[i-1,j+1])/16 + 
                           np.int(R_new[i,j-1])/8 + np.int(R_new[i,j])/4 + np.int(R_new[i,j+1])/8 +
                           np.int(R_new[i+1,j-1])/16 + np.int(R_new[i+1,j])/8 + np.int(R_new[i+1,j+1])/16))
    
    for i in range(1,95):
        for j in range(1,5):
            matrix[i,j] = np.uint8((np.int(R_new[i-1,j-1])/16 + np.int(R_new[i-1,j])/8 + np.int(R_new[i-1,j+1])/16 + 
                           np.int(R_new[i,j-1])/8 + np.int(R_new[i,j])/4 + np.int(R_new[i,j+1])/8 +
                           np.int(R_new[i+1,j-1])/16 + np.int(R_new[i+1,j])/8 + np.int(R_new[i+1,j+1])/16))
    
    for i in range(1,95):
        for j in range(91,95):
            matrix[i,j] = np.uint8((np.int(R_new[i-1,j-1])/16 + np.int(R_new[i-1,j])/8 + np.int(R_new[i-1,j+1])/16 + 
                           np.int(R_new[i,j-1])/8 + np.int(R_new[i,j])/4 + np.int(R_new[i,j+1])/8 +
                           np.int(R_new[i+1,j-1])/16 + np.int(R_new[i+1,j])/8 + np.int(R_new[i+1,j+1])/16))
    
    #High-pass sharpening
    for i in range(4,91):
        for j in range(4,91):
            matrix[i,j] = np.uint8(1/8*(np.int(R_new[i-1,j-1])*-1 + np.int(R_new[i-1,j])*-1 + np.int(R_new[i-1,j+1])*-1 + 
                           np.int(R_new[i,j-1])*-1 + np.int(R_new[i,j])*16 + np.int(R_new[i,j+1])*-1 +
                           np.int(R_new[i+1,j-1])*-1 + np.int(R_new[i+1,j])*-1 + np.int(R_new[i+1,j+1])*-1))
    
    #plt.figure(figsize=(10,10))
    final = matrix[1:95,1:95]
    #plt.imshow(final) 
    return final


#Weighted Averaging filter (Gassian Filter)
def weighted_filter(color, b):
    matrix = np.zeros((96,96), dtype=np.uint8)
    R_new = color
    for i in range(1,95):
        for j in range(1,95):
            matrix[i,j] = np.uint8(1/(1+b)/(1+b)*(np.int(R_new[i-1,j-1])*1 + np.int(R_new[i-1,j])*b + np.int(R_new[i-1,j+1])*1 + 
                           np.int(R_new[i,j-1])*b + np.int(R_new[i,j])*b*b + np.int(R_new[i,j+1])*b +
                           np.int(R_new[i+1,j-1])*1 + np.int(R_new[i+1,j])*b + np.int(R_new[i+1,j+1])*1))
    #plt.figure(figsize=(10,10))
    final = matrix[1:95,1:95]
    #plt.imshow(final) 
    return final
    

#Brightness/Contrast adjustment
def bright_contrast(input_img): 
    b,g,r = cv2.split(input_img)
    #Resize to be 94*94 
    b = b[1:95,1:95]
    g = g[1:95,1:95]
    r = r[1:95,1:95]
    rgb_img = cv2.merge([r,g,b])
    RANDOM_BRIGHTNESS = 64  # range (0-100), 0=no change
    RANDOM_CONTRAST = 7   # range (0-100), 0=no change
    
    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = np.uint8(rgb_img + br)
        
    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = np.uint8(rgb_img * cr)
    
    #plt.figure(figsize=(10,10))
    #plt.imshow(rgb_img)
    return rgb_img
       

def apply(img, function):
    R_initial = img[:,:,0]
    G_initial = img[:,:,1]
    B_initial = img[:,:,2]
    
    #Apply function
    if (function == 'weighted_filter'):
        R_final = weighted_filter(R_initial, 2)
        G_final = weighted_filter(G_initial, 2)
        B_final = weighted_filter(B_initial, 2)
    elif (function == 'blur_sharp'):
        R_final = blur_sharp(R_initial)
        G_final = blur_sharp(G_initial)
        B_final = blur_sharp(B_initial)
    elif (function == 'bright_contrast'):
        img_final = bright_contrast(img)
        return img_final
    
    img_final = np.dstack((R_final, G_final))
    img_final = np.dstack((img_final, B_final))
    plt.figure(figsize=(10,10))
    plt.imshow(img_final)
    return img_final

### Test curx functions above 
def main():
    img = cv2.imread('histopathologic-cancer-detection/train/d42e09bc5560bb88ef86b34f58e0657381455fa2.tif')
    case1 = apply(img, 'weighted_filter')
    case2 = apply(img, 'blur_sharp')
    case3 = apply(img, 'bright_contrast')

main()