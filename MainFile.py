#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
from skimage.feature import greycomatrix, greycoprops

#====================== 1.READ A INPUT IMAGE =========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image')
plt.axis ('off')
plt.show()

#============================ 2.IMAGE PREPROCESSING ====================

#    glcm = greycomatrix(patch.astype(int), [5], [0], 256, symmetric=True, normed=True)

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====

try:            
    gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray11 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray11)
plt.axis ('off')
plt.show()


#===== GAUSSIAN BLUR ====

Gaussian = cv2.GaussianBlur(img, (7, 7), 0)

plt.imshow(Gaussian)
plt.title('Gaussian Blur')
plt.show()


#============================ 3.FEATURE EXTRACTION ====================

# === MEAN MEDIAN VARIANCE ===

mean_val = np.mean(gray11)
median_val = np.median(gray11)
var_val = np.var(gray11)
Test_features = [mean_val,median_val,var_val]


print()
print("----------------------------------------------")
print("FEATURE EXTRACTION --> MEAN, VARIANCE, MEDIAN ")
print("----------------------------------------------")
print()
print("1. Mean Value     =", mean_val)
print()
print("2. Median Value   =", median_val)
print()
print("1. Variance Value =", var_val)


# === GRAY LEVEL CO OCCURENCE MATRIX ===

# === GRAY LEVEL CO OCCURENCE MATRIX ===

print()
print("-----------------------------------------------------")
print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
print("-----------------------------------------------------")
print()


PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
plt.show()

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.show()


sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print()
print("GLCM FEATURES =")
print()
print(Glcm_fea)

#
#============================ 5. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split

mild_data = os.listdir('Dataset/Mild/')
mod_data = os.listdir('Dataset/Moderate/')
no_data = os.listdir('Dataset/NO_DR/')
pro_data = os.listdir('Dataset/Proliferate_DR/')
severe_data = os.listdir('Dataset/Severe/')


###############
#       
dot1= []
labels1 = [] 
for img11 in mild_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/Mild//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in mod_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/Moderate//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)

for img11 in no_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/NO_DR//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)

for img11 in pro_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/Proliferate_DR//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(4)


for img11 in severe_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/Severe//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(5)

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))


#=============================== CLASSIFICATION =================================

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


# ======== CNN ===========
    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout




# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(6,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

pred_cnn = model.predict([x_train2])

y_pred2 = pred_cnn.reshape(-1)
y_pred2[y_pred2<0.5] = 0
y_pred2[y_pred2>=0.5] = 1
y_pred2 = y_pred2.astype('int')


print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()
acc_cnn=accuracy[1]*100
print("1. Accuracy   =", acc_cnn,'%')
print()
print("2. Error Rate =",100-acc_cnn)



# ================== K NEAREST NEIGHBOUR ======================

from keras.utils import to_categorical

x_train11=np.zeros((len(x_train),50))
for i in range(0,len(x_train)):
        x_train11[i,:]=np.mean(x_train[i])

x_test11=np.zeros((len(x_test),50))
for i in range(0,len(x_test)):
        x_test11[i,:]=np.mean(x_test[i])


y_train11=np.array(y_train)
y_test11=np.array(y_test)

train_Y_one_hot = to_categorical(y_train11)
test_Y_one_hot = to_categorical(y_test)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier() 


knn.fit(x_train11,y_train11)


y_pred = knn.predict(x_train11)

from sklearn import metrics

accuracy_test=metrics.accuracy_score(y_pred,y_train11)*100

accuracy_train=metrics.accuracy_score(y_train11,y_train11)*100

acc_overall_knn=(accuracy_test + accuracy_train)/2


print("-------------------------------------")
print("PERFORMANCE ---------> (KNN)")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_overall_knn,'%')
print()
print("2. Error Rate =",100-acc_overall_knn)


#=============================== PREDICTION =================================

print()
print("-----------------------")
print("       PREDICTION      ")
print("-----------------------")
print()


Total_length = len(mild_data) + len(mod_data) + len(no_data) + len(pro_data) + len(severe_data)
 

temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('------------------------')
    print(' DIABETIC ---> MILD     ')
    print('------------------------')

elif labels1[zz[0][0]] == 2:
    print('-----------------------')
    print(' DIABETIC ---> MODERATE')
    print('-----------------------')
elif labels1[zz[0][0]] == 3:
    print('-----------------------')
    print(' No DIABETIC')
    print('-----------------------')
elif labels1[zz[0][0]] == 4:
    print('--------------------------')
    print(' DIABETIC ---> PROLIFERATE')
    print('--------------------------')
elif labels1[zz[0][0]] == 5:
    print('-----------------------')
    print(' DIABETIC ---> SEVERE  ')
    print('-----------------------')

#=============================== VISUALIZATIOn =================================


print()
print("-----------------------------------------------------------------------")
print()


import matplotlib.pyplot as plt
vals=[acc_cnn,acc_overall_knn]
inds=range(len(vals))
labels=["CNN ","KNN "]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison graph --> ACCURACY')
plt.show() 

