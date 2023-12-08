# Relatório de Atividades PDI - 2023

# Giovana Perazzolo Menato
## Aula 1 - 31/07: Apresentação da Discipina

- Apresentação da disciplina
- Critérios de avaliação
- Introdução ao processamento digital de imagens
## Aula 2 - 07/08: Fundamentos da Imagem Digital

- Elementos da percepção visual humana
- Sensores para aquisição de imagens
- Amostragem e quantização
- Exemplos Python 
### Código Python e Imagem: Iniciais do Nome
import numpy as np
import matplotlib.pyplot as plt

colunas = 25
linhas = 10
image_matrix = np.zeros([linhas, colunas])
print(image_matrix.shape)

#fazendo G
image_matrix[1:7,1]=255 #descendo vertical
image_matrix[1,2:5]=255 #indo para o lado horizontal
image_matrix[6,2:5]=255
image_matrix[4:6,4:5]=255
image_matrix[4,3:5]=255

#fazendo P
image_matrix[1:7,7]=255
image_matrix[1,7:11]=255
image_matrix[1:4,11]=255
image_matrix[4,7:12]=255

#fazendo M
image_matrix[1:7,14]=255
image_matrix[1:3,15]=255
image_matrix[3:5,16]=255
image_matrix[5:7,17]=255
image_matrix[3:5,18]=255
image_matrix[1:3,19]=255
image_matrix[1:7,20]=255


plt.imshow(image_matrix, cmap='gray')
plt.show()

#todas as colunas: [0,:]
#intervalo entre linhas [3:5,:]

## Aula 3 - 14/08: Vizinhança e Operações em Imagens

- Relacionamentos básicos entre pixels
- Operações espaciais
    - Transformações geométricas
    - Escala
    - Rotação
    - Translação
    - Cisalhamento
- Transformações de intensidade
### Código exemplo da Aula
import numpy as np
import matplotlib.pyplot as plt

width = 10
height = 5
image_matrix = np.zeros([height, width])
print(image_matrix.shape)

#Acessando um pixel
image_matrix[0,0] = 255
#Acessando uma linha (Primeira linha da matrix)
image_matrix[0,:] = 255
#Acessando mais de uma linha (Quarta e quinta linha da matrix)
image_matrix[3:5,:] = 255
#Acessando uma coluna (Ultima colua)
image_matrix[:,9] = 120

print(image_matrix)
plt.imshow(image_matrix, cmap='gray')
plt.show()
### Código Exercício em Aula: Salt and Pepper
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_pillow = Image.open('img/lena_gray_512_salt_pepper.tif')
f_image_nd = np.array(image_pillow)
g_image_nd = np.zeros(f_image_nd.shape)

#neighborhood operation (operação por vizinhança)
l = f_image_nd.shape[0]
c = f_image_nd.shape[1]
k = 2
#print("Imagem")

print(f_image_nd[0:5,0:5])
for x in range(k, l-k): #linhas
    for y in range(k, c-k): #colunas
        s_xy = f_image_nd[x-k:x+k+1,y-k:y+k+1]
        g_image_nd[x,y] = np.median(s_xy).astype(int)
        #print('janela')
        #print(s_xy)

    
#create two columns plot
fig = plt.figure()
plt1 = plt.subplot(1,2,1)
plt2 = plt.subplot(1,2,2)
plt1.title.set_text('Original Image')
plt2.title.set_text('Filtred Image')

plt1.imshow(f_image_nd, cmap='gray')
plt2.imshow(g_image_nd, cmap='gray', vmin=0, vmax=500)
plt.show()
### Código Exercício em Aula: Lena
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('img/lena.jpg')
im.show()

im_ndarray = np.array(im)
implot = plt.imshow(im_ndarray, cmap='gray')
plt.show()

#operação pixel a pixel (escurecer imagem)
im_dark = im_ndarray.copy()
im_dark = im_dark/4

#create two columns plot
fig = plt.figure()
plt1 = plt.subplot(1,2,1)
plt2 = plt.subplot(1,2,2)
plt1.title.set_text('Original Image')
plt2.title.set_text('Filtred Image')

plt1.imshow(im_ndarray, cmap='gray')
plt2.imshow(im_dark, cmap='gray', vmin=0, vmax=500)
plt.show()
## Aula 4 - 21/08: Fundamentos Imagens

1. [OPERAÇÃO PONTO A PONTO]:
- Calcular o negativo das imagens;
- Diminuir pela metade a intensidade dos pixels;
- Incluir 4 quadrados brancos 10 x 10 pixels em cada canto das imagens;
- Incluir 1 quadrado preto 15X15 no centro das imagens

2. [OPERAÇÃO POR VIZINHANÇA]: Utilizar kernel 3x3 pixels e desconsiderar pixels das extremidades. Para cada filtro implementar utilizando apenas numpy, utilizando pillow, utilizando opencv e utilizando scipy.
- Calcular o filtro da média;
- Calcular o filtro da mediana;

3. [TRANSFORMAÇÕES GEOMÉTRICAS]: Para cada filtro implementar utilizando apenas numpy, utilizando pillow, utilizando opencv e utilizando scipy.
- Escala: Redução em 1.5x e aumentar em 2.5x;
- Rotação em 45º, 90º e 100º;
- Translação utilizar os parâmetros que quiser nas coordenadas x e y;
- Translação em 35 pixel no eixo X, 45 eixo Y; 



### Exercício 1:
Calcular Negativo
import numpy as np
from PIL import Image  
from numpy import asarray
import matplotlib.pyplot as plt

# Open image
imageLena = Image.open('img/lena.jpg')
imageHouse = Image.open('img/house.tif')
imageCamera = Image.open('img/cameraman.tif')

# convert image to numpy array    
npImageLena = np.array(imageLena) 
npImageHouse = np.array(imageHouse) 
npImageCamera = np.array(imageCamera) 
# Create negative image
npImageNegativeLena = np.array(npImageLena) 
npImageNegativeLena = 255 - npImageNegativeLena;
npImageNegativeHouse = np.array(npImageHouse) 
npImageNegativeHouse = 255 - npImageNegativeHouse;
npImageNegativeCamera = np.array(npImageCamera) 
npImageNegativeCamera = 255 - npImageNegativeCamera;
# Display images and their negatives using Matplotlib
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Display original images
axs[0, 0].imshow(npImageLena, cmap='gray')
axs[0, 0].set_title('Original Lena')
axs[0, 0].axis('off')

axs[1, 0].imshow(npImageHouse, cmap='gray')
axs[1, 0].set_title('Original House')
axs[1, 0].axis('off')

axs[2, 0].imshow(npImageCamera, cmap='gray')
axs[2, 0].set_title('Original Camera')
axs[2, 0].axis('off')

# Display negative images
axs[0, 1].imshow(npImageNegativeLena, cmap='gray')
axs[0, 1].set_title('Negative Lena')
axs[0, 1].axis('off')

axs[1, 1].imshow(npImageNegativeHouse, cmap='gray')
axs[1, 1].set_title('Negative House')
axs[1, 1].axis('off')

axs[2, 1].imshow(npImageNegativeCamera, cmap='gray')
axs[2, 1].set_title('Negative Camera')
axs[2, 1].axis('off')

plt.tight_layout()
plt.show()
Diminuir pela Metade a Intensidade dos Pixels
import numpy as np
from PIL import Image  
from numpy import asarray
import matplotlib.pyplot as plt

# Open image
imageLena = Image.open('img/lena.jpg')
imageHouse = Image.open('img/house.tif')
imageCamera = Image.open('img/cameraman.tif')

# convert image to numpy array    
npImageLena = np.array(imageLena) 
npImageHouse = np.array(imageHouse) 
npImageCamera = np.array(imageCamera) 

# divide by 2 pixels
npImageLena =  (npImageLena / 2).astype(int);
npImageHouse =  (npImageHouse / 2).astype(int);
npImageCamera =  (npImageCamera / 2).astype(int);

print(npImageLena.shape)
print(npImageHouse.shape)
print(npImageCamera.shape)
Incluir 4 quadrados brancos 10 x 10 pixels em cada canto das imagens
imageLena = Image.open('img/lena.jpg')
imageHouse = Image.open('img/house.tif')
imageCamera = Image.open('img/cameraman.tif')

# convert image to numpy array    
npImageLena = np.array(imageLena) 
npImageHouse = np.array(imageHouse) 
npImageCamera = np.array(imageCamera) 
# Add white squares to the corners of the Camera image
npImageCamera[0:11,0:11] = 255
npImageCamera[501:512, 501:512] = 255
npImageCamera[0:11, 501:512] = 255
npImageCamera[501:512, 0:11] = 255

# Add white squares to the corners of the Lena image
npImageLena[0:11,0:11] = 255
npImageLena[589:600, 589:600] = 255
npImageLena[0:11, 589:600] = 255
npImageLena[589:600, 0:11] = 255

# Add white squares to the corners of the House image
npImageHouse[0:11,0:11] = 255
npImageHouse[589:600, 589:600] = 255
npImageHouse[0:11, 589:600] = 255
npImageHouse[589:600, 0:11] = 255

# Display the images using Matplotlib
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display Camera image with white squares
axs[0].imshow(npImageCamera, cmap='gray')
axs[0].set_title('Camera with Squares')
axs[0].axis('off')

# Display Lena image with white squares
axs[1].imshow(npImageLena, cmap='gray')
axs[1].set_title('Lena with Squares')
axs[1].axis('off')

# Display House image with white squares
axs[2].imshow(npImageHouse, cmap='gray')
axs[2].set_title('House with Squares')
axs[2].axis('off')

plt.tight_layout()
plt.show()
Incluir um quadrado preto 15x15 no centro das imagens
imageLena = Image.open('img/lena.jpg')
imageHouse = Image.open('img/house.tif')
imageCamera = Image.open('img/cameraman.tif')

# convert image to numpy array    
npImageLena = np.array(imageLena) 
npImageHouse = np.array(imageHouse) 
npImageCamera = np.array(imageCamera) 

npImageLena =  (npImageLena / 2).astype(int);
npImageHouse =  (npImageHouse / 2).astype(int);
npImageCamera =  (npImageCamera / 2).astype(int);

npImageLena[142:157,142:157] = 0
npImageCamera[248:263,248:263] = 0
npImageHouse[292:307,292:307] = 0

# Display the images using Matplotlib
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Display Camera image with white squares
axs[0].imshow(npImageCamera, cmap='gray')
axs[0].set_title('Camera with Squares black')
axs[0].axis('off')

# Display Lena image with white squares
axs[1].imshow(npImageLena, cmap='gray')
axs[1].set_title('Lena with Squares black')
axs[1].axis('off')

# Display House image with white squares
axs[2].imshow(npImageHouse, cmap='gray')
axs[2].set_title('House with Squares black')
axs[2].axis('off')

plt.tight_layout()
plt.show()
### Exercício 2
Calcular o filtro da média
# Load images using Pillow
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = np.array(Image.open(lena_path).convert('L'))
cam = np.array(Image.open(cam_path).convert('L'))
house = np.array(Image.open(house_path).convert('L'))

def apply_neighborhood_operation_mean(image, k=3):
    output_image = np.zeros(image.shape)
    l, c = image.shape
    for x in range(k, l-k):
        for y in range(k, c-k):
            s_xy = image[x-k:x+k+1, y-k:y+k+1]
            output_image[x, y] = np.mean(s_xy).astype(int)
    return output_image

def apply_neighborhood_operation_median(image, k=3):
    output_image = np.zeros(image.shape)
    l, c = image.shape
    for x in range(k, l-k):
        for y in range(k, c-k):
            s_xy = image[x-k:x+k+1, y-k:y+k+1]
            output_image[x, y] = np.median(s_xy).astype(int)
    return output_image

# Apply neighborhood operation (mean) to each image
g_image_ndLena_mean = apply_neighborhood_operation_mean(lena)
g_image_ndCam_mean = apply_neighborhood_operation_mean(cam)
g_image_ndHouse_mean = apply_neighborhood_operation_mean(house)

# Apply neighborhood operation (median) to each image
g_image_ndLena_median = apply_neighborhood_operation_median(lena)
g_image_ndCam_median = apply_neighborhood_operation_median(cam)
g_image_ndHouse_median = apply_neighborhood_operation_median(house)

# Display images
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Display Lena images
axs[0, 0].imshow(lena, cmap='gray')
axs[0, 0].set_title('Original Lena')
axs[0, 1].imshow(g_image_ndLena_mean, cmap='gray')
axs[0, 1].set_title('Lena (Mean Filter)')
axs[0, 2].imshow(g_image_ndLena_median, cmap='gray')
axs[0, 2].set_title('Lena (Median Filter)')

# Display Cameraman images
axs[1, 0].imshow(cam, cmap='gray')
axs[1, 0].set_title('Original Cameraman')
axs[1, 1].imshow(g_image_ndCam_mean, cmap='gray')
axs[1, 1].set_title('Cameraman (Mean Filter)')
axs[1, 2].imshow(g_image_ndCam_median, cmap='gray')
axs[1, 2].set_title('Cameraman (Median Filter)')

# Display House images
axs[2, 0].imshow(house, cmap='gray')
axs[2, 0].set_title('Original House')
axs[2, 1].imshow(g_image_ndHouse_mean, cmap='gray')
axs[2, 1].set_title('House (Mean Filter)')
axs[2, 2].imshow(g_image_ndHouse_median, cmap='gray')
axs[2, 2].set_title('House (Median Filter)')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
Incorporando as Imagens pelo Pillow
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_neighborhood_operation(img, operation='mean'):
    width, height = img.size
    new_img = Image.new('L', (width, height))
    pixels = img.load()
    new_pixels = new_img.load()

    for x in range(1, width-1):
        for y in range(1, height-1):
            # Extract 3x3 neighborhood
            neighborhood = [
                pixels[x-1, y-1], pixels[x, y-1], pixels[x+1, y-1],
                pixels[x-1, y], pixels[x, y], pixels[x+1, y],
                pixels[x-1, y+1], pixels[x, y+1], pixels[x+1, y+1]
            ]
            
            if operation == 'mean':
                new_pixels[x, y] = sum(neighborhood) // 9
            elif operation == 'median':
                new_pixels[x, y] = sorted(neighborhood)[4]

    return new_img

# Load images
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = Image.open(lena_path).convert('L')
cam = Image.open(cam_path).convert('L')
house = Image.open(house_path).convert('L')

# Apply neighborhood operation
lena_mean = apply_neighborhood_operation(lena, 'mean')
cam_mean = apply_neighborhood_operation(cam, 'mean')
house_mean = apply_neighborhood_operation(house, 'mean')

lena_median = apply_neighborhood_operation(lena, 'median')
cam_median = apply_neighborhood_operation(cam, 'median')
house_median = apply_neighborhood_operation(house, 'median')

# Display images
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Display Lena images
axs[0, 0].imshow(lena, cmap='gray')
axs[0, 0].set_title('Original Lena')
axs[0, 1].imshow(lena_mean, cmap='gray')
axs[0, 1].set_title('Lena (Mean Filter)')
axs[0, 2].imshow(lena_median, cmap='gray')
axs[0, 2].set_title('Lena (Median Filter)')

# Display Cameraman images
axs[1, 0].imshow(cam, cmap='gray')
axs[1, 0].set_title('Original Cameraman')
axs[1, 1].imshow(cam_mean, cmap='gray')
axs[1, 1].set_title('Cameraman (Mean Filter)')
axs[1, 2].imshow(cam_median, cmap='gray')
axs[1, 2].set_title('Cameraman (Median Filter)')

# Display House images
axs[2, 0].imshow(house, cmap='gray')
axs[2, 0].set_title('Original House')
axs[2, 1].imshow(house_mean, cmap='gray')
axs[2, 1].set_title('House (Mean Filter)')
axs[2, 2].imshow(house_median, cmap='gray')
axs[2, 2].set_title('House (Median Filter)')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()


Incorporando as Imagens pelo OpenCV
import cv2
import matplotlib.pyplot as plt

# Load images
lena_path = 'img/lena.jpg'
house_path = 'img/house.tif'
cam_path = 'img/cameraman.tif'

lena = cv2.imread(lena_path, cv2.IMREAD_GRAYSCALE)
house = cv2.imread(house_path, cv2.IMREAD_GRAYSCALE)
cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

# Apply mean and median filtering using 3x3 kernel
lena_mean = cv2.blur(lena, (3,3))
lena_median = cv2.medianBlur(lena, 3)

house_mean = cv2.blur(house, (3,3))
house_median = cv2.medianBlur(house, 3)

cam_mean = cv2.blur(cam, (3,3))
cam_median = cv2.medianBlur(cam, 3)

# Display images
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

axs[0, 0].imshow(lena, cmap='gray')
axs[0, 0].set_title('Original Lena')
axs[0, 1].imshow(lena_mean, cmap='gray')
axs[0, 1].set_title('Lena Mean Filtered')
axs[0, 2].imshow(lena_median, cmap='gray')
axs[0, 2].set_title('Lena Median Filtered')

axs[1, 0].imshow(house, cmap='gray')
axs[1, 0].set_title('Original House')
axs[1, 1].imshow(house_mean, cmap='gray')
axs[1, 1].set_title('House Mean Filtered')
axs[1, 2].imshow(house_median, cmap='gray')
axs[1, 2].set_title('House Median Filtered')

axs[2, 0].imshow(cam, cmap='gray')
axs[2, 0].set_title('Original Cameraman')
axs[2, 1].imshow(cam_mean, cmap='gray')
axs[2, 1].set_title('Cameraman Mean Filtered')
axs[2, 2].imshow(cam_median, cmap='gray')
axs[2, 2].set_title('Cameraman Median Filtered')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
Incorporando as imagens pelo Scipy
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# Load images
lena_path = 'img/lena.jpg'
house_path = 'img/house.tif'
cam_path = 'img/cameraman.tif'

lena = cv2.imread(lena_path, cv2.IMREAD_GRAYSCALE)
house = cv2.imread(house_path, cv2.IMREAD_GRAYSCALE)
cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)

# Apply mean filtering using 3x3 kernel
kernel = [[1/9, 1/9, 1/9],
          [1/9, 1/9, 1/9],
          [1/9, 1/9, 1/9]]

lena_mean = ndimage.convolve(lena, kernel)
house_mean = ndimage.convolve(house, kernel)
cam_mean = ndimage.convolve(cam, kernel)

# Apply median filtering using 3x3 kernel
lena_median = ndimage.median_filter(lena, size=3)
house_median = ndimage.median_filter(house, size=3)
cam_median = ndimage.median_filter(cam, size=3)

# Display images
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

axs[0, 0].imshow(lena, cmap='gray')
axs[0, 0].set_title('Original Lena')
axs[0, 1].imshow(lena_mean, cmap='gray')
axs[0, 1].set_title('Lena Mean Filtered')
axs[0, 2].imshow(lena_median, cmap='gray')
axs[0, 2].set_title('Lena Median Filtered')

axs[1, 0].imshow(house, cmap='gray')
axs[1, 0].set_title('Original House')
axs[1, 1].imshow(house_mean, cmap='gray')
axs[1, 1].set_title('House Mean Filtered')
axs[1, 2].imshow(house_median, cmap='gray')
axs[1, 2].set_title('House Median Filtered')

axs[2, 0].imshow(cam, cmap='gray')
axs[2, 0].set_title('Original Cameraman')
axs[2, 1].imshow(cam_mean, cmap='gray')
axs[2, 1].set_title('Cameraman Mean Filtered')
axs[2, 2].imshow(cam_median, cmap='gray')
axs[2, 2].set_title('Cameraman Median Filtered')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
### Exercício 3
- Escala: Redução em 1.5x e aumentar em 2.5x;
- Rotação em 45º, 90º e 100º;
import numpy as np
from numpy import asarray
from PIL import Image
from scipy import ndimage

def apply_transforms(image_path, title):
    # Open image
    image_in = Image.open(image_path)
    
    # Convert image to numpy array    
    image_np = np.array(image_in) 

    # Zoom or Shrink image
    image_np_zoom = ndimage.zoom(image_np, (2.5, 2.5))
        
    # Rotation image 45º
    image_np_rotate  = ndimage.rotate(image_np, -100, cval=128)

    # Shear image
    height, width = image_np.shape
    transform = [[1, 0, 0],
                 [0.5, 1, 0],
                 [0, 0, 1]]
    image_np_shear = ndimage.affine_transform(image_np,
                                         transform,
                                         offset=(0, -height//2, 0),
                                         output_shape=(height, width+height//2))
    
    # Display images using matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_np, cmap='gray')
    axs[0].set_title(f'Original {title}')
    axs[1].imshow(image_np_zoom, cmap='gray')
    axs[1].set_title(f'Zoomed {title}')
    axs[2].imshow(image_np_rotate, cmap='gray')
    axs[2].set_title(f'Rotated {title}')
    axs[3].imshow(image_np_shear, cmap='gray')
    axs[3].set_title(f'Sheared {title}')
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Paths to the images
image_pathLena = 'img/lena.jpg'
image_pathHouse = 'img/house.tif'
image_pathCamera = 'img/cameraman.tif'

# Apply transformations to each image
apply_transforms(image_pathLena, 'Lena')
apply_transforms(image_pathHouse, 'House')
apply_transforms(image_pathCamera, 'CameraMan')
- Translação utilizar os parâmetros que quiser nas coordenadas x e y;
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

# Load the images
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = np.array(Image.open(lena_path).convert('L'))  # Convert to grayscale
cam = np.array(Image.open(cam_path).convert('L'))
house = np.array(Image.open(house_path).convert('L'))

# Copy the images
translLena = lena.copy()
translCam = cam.copy()
translHouse = house.copy()

# Define the translation (shift) vector
shift_x = 50  
shift_y = -30
shift_vector = [shift_y, shift_x]

# Apply the translation to each image
translated_image_lena = ndimage.shift(translLena, shift_vector, mode='constant', cval=0)
translated_image_cam = ndimage.shift(translCam , shift_vector, mode='constant', cval=0)
translated_image_house = ndimage.shift(translHouse, shift_vector, mode='constant', cval=0)

# Display the translated images using matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=50 y=-30")
plt2.title.set_text("Cameraman Translação x=50 y=-30")
plt3.title.set_text("House Translação x=50 y=-30")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
- Translação em 35 pixel no eixo X, 45 eixo Y;
# Load the images
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = np.array(Image.open(lena_path).convert('L'))  # Convert to grayscale
cam = np.array(Image.open(cam_path).convert('L'))
house = np.array(Image.open(house_path).convert('L'))

# Copy the images
translLena = lena.copy()
translCam = cam.copy()
translHouse = house.copy()

# Define the translation (shift) vector
shift_x = 35  
shift_y = 45
shift_vector = [shift_y, shift_x]

# Apply the translation to each image
translated_image_lena = ndimage.shift(translLena, shift_vector, mode='constant', cval=0)
translated_image_cam = ndimage.shift(translCam , shift_vector, mode='constant', cval=0)
translated_image_house = ndimage.shift(translHouse, shift_vector, mode='constant', cval=0)

# Display the translated images using matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=35 y=45")
plt2.title.set_text("Cameraman Translação x=35 y=45")
plt3.title.set_text("House Translação x=35 y=45")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
Utilizando a biblioteca Pillow 
- Redução em 1.5x e aumento de 2.5x
- Rotação de 45°, 90° e 100°
import PIL
import numpy as np
import matplotlib.pyplot as plt

def apply_transforms(image_path, title):
    # Open image
    image_in = Image.open(image_path)
    
    # Zoom or Shrink image
    size = tuple(int(dim * 2.5) for dim in image_in.size)
    image_zoom = image_in.resize(size, PIL.Image.LANCZOS)
        
    # Rotation image 45º
    image_rotate = image_in.rotate(100, resample=Image.BICUBIC, fillcolor=128)

    # Shear image
    width, height = image_in.size
    m = -0.5  # Shear factor
    shear_matrix = (1, m, -m*height/2, 0, 1, 0)
    image_shear = image_in.transform((width + int(m*height), height), Image.AFFINE, shear_matrix, Image.BICUBIC)
    
    # Display images using matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_in, cmap='gray')
    axs[0].set_title(f'Original {title}')
    axs[1].imshow(image_zoom, cmap='gray')
    axs[1].set_title(f'Zoomed {title}')
    axs[2].imshow(image_rotate, cmap='gray')
    axs[2].set_title(f'Rotated {title}')
    axs[3].imshow(image_shear, cmap='gray')
    axs[3].set_title(f'Sheared {title}')
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Paths to the images
image_pathLena = 'img/lena.jpg'
image_pathHouse = 'img/house.tif'
image_pathCamera = 'img/cameraman.tif'

# Apply transformations to each image
apply_transforms(image_pathLena, 'Lena')
apply_transforms(image_pathHouse, 'House')
apply_transforms(image_pathCamera, 'CameraMan')
- transalação utilizando os parâmetros que quiser
# Load the images
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = Image.open(lena_path).convert('L')  # Convert to grayscale
cam = Image.open(cam_path).convert('L')
house = Image.open(house_path).convert('L')

# Define the translation (shift) vector
shift_x = 50  
shift_y = -30

# Apply the translation to each image
translated_image_lena = lena.transform(lena.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)
translated_image_cam = cam.transform(cam.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)
translated_image_house = house.transform(house.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)

# Display the translated images using matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=50 y=-30")
plt2.title.set_text("Cameraman Translação x=50 y=-30")
plt3.title.set_text("House Translação x=50 y=-30")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
- translação em 35 no eixo x e 45 no eixo y
# Load the images
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

lena = Image.open(lena_path).convert('L')  # Convert to grayscale
cam = Image.open(cam_path).convert('L')
house = Image.open(house_path).convert('L')

# Define the translation (shift) vector
shift_x = 35  
shift_y = 45

# Apply the translation to each image
translated_image_lena = lena.transform(lena.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)
translated_image_cam = cam.transform(cam.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)
translated_image_house = house.transform(house.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=0)

# Display the translated images using matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=35 y=45")
plt2.title.set_text("Cameraman Translação x=35 y=45")
plt3.title.set_text("House Translação x=35 y=45")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
Utilizando a biblioteca OpenCV
- Redução de 1.5x e aumento de 2.5x
- Rotação de 45°, 90° e 100°
import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_transforms(image_path, title):
    # Open image
    image_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Zoom or Shrink image
    zoom_scale = 2.5
    image_zoom = cv2.resize(image_in, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)
        
    # Rotation image 45º
    rows, cols = image_in.shape
    M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), -100, 1)
    image_rotate = cv2.warpAffine(image_in, M_rotate, (cols, rows), borderValue=128)
    
    # Shear image
    M_shear = np.float32([[1, 0.5, -0.5*cols/2], [0, 1, 0]])
    image_shear = cv2.warpAffine(image_in, M_shear, (int(cols + 0.5*rows), rows), borderValue=128)
    
    # Display images using matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_in, cmap='gray')
    axs[0].set_title(f'Original {title}')
    axs[1].imshow(image_zoom, cmap='gray')
    axs[1].set_title(f'Zoomed {title}')
    axs[2].imshow(image_rotate, cmap='gray')
    axs[2].set_title(f'Rotated {title}')
    axs[3].imshow(image_shear, cmap='gray')
    axs[3].set_title(f'Sheared {title}')
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Paths to the images
image_pathLena = 'img/lena.jpg'
image_pathHouse = 'img/house.tif'
image_pathCamera = 'img/cameraman.tif'

# Apply transformations to each image
apply_transforms(image_pathLena, 'Lena')
apply_transforms(image_pathHouse, 'House')
apply_transforms(image_pathCamera, 'CameraMan')
- translação utilizando os parâmetros que quiser
def apply_translation(image_path, shift_x, shift_y, title):
    # Carregar a imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Definir a matriz de translação
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Aplicar a translação
    translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return translated_image

# Caminhos para as imagens
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

# Definir o vetor de translação
shift_x = 50  
shift_y = -30

# Aplicar a translação a cada imagem
translated_image_lena = apply_translation(lena_path, shift_x, shift_y, 'Lena')
translated_image_cam = apply_translation(cam_path, shift_x, shift_y, 'Cameraman')
translated_image_house = apply_translation(house_path, shift_x, shift_y, 'House')

# Exibir as imagens traduzidas usando matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=50 y=-30")
plt2.title.set_text("Cameraman Translação x=50 y=-30")
plt3.title.set_text("House Translação x=50 y=-30")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
- translação em 35 no eixo x e 45 no y
def apply_translation(image_path, shift_x, shift_y):
    # Carregar a imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Definir a matriz de translação
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Aplicar a translação
    translated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return translated_image

# Caminhos para as imagens
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

# Definir o vetor de translação
shift_x = 35  
shift_y = 45

# Aplicar a translação a cada imagem
translated_image_lena = apply_translation(lena_path, shift_x, shift_y)
translated_image_cam = apply_translation(cam_path, shift_x, shift_y)
translated_image_house = apply_translation(house_path, shift_x, shift_y)

# Exibir as imagens traduzidas usando matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=35 y=45")
plt2.title.set_text("Cameraman Translação x=35 y=45")
plt3.title.set_text("House Translação x=35 y=45")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
Utilizando a biblioteca Scipy

import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

def apply_transforms(image_path, title):
    # Open image
    image_in = imageio.imread(image_path)
    
    # Convert to grayscale if the image is RGB
    if image_in.ndim == 3:
        image_in = imageio.imread(image_path, as_gray=True)
    
    # Zoom or Shrink image
    image_np_zoom = ndimage.zoom(image_in, 2.5)
        
    # Rotation image 45º
    image_np_rotate  = ndimage.rotate(image_in, -100, cval=128)

    # Shear image
    height, width = image_in.shape
    transform = [[1, 0, 0],
                 [0.5, 1, 0],
                 [0, 0, 1]]
    image_np_shear = ndimage.affine_transform(image_in,
                                         transform,
                                         offset=(0, -height//2),
                                         output_shape=(height, width+height//2))
    
    # Display images using matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image_in, cmap='gray')
    axs[0].set_title(f'Original {title}')
    axs[1].imshow(image_np_zoom, cmap='gray')
    axs[1].set_title(f'Zoomed {title}')
    axs[2].imshow(image_np_rotate, cmap='gray')
    axs[2].set_title(f'Rotated {title}')
    axs[3].imshow(image_np_shear, cmap='gray')
    axs[3].set_title(f'Sheared {title}')
    
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Paths to the images
image_pathLena = 'img/lena.jpg'
image_pathHouse = 'img/house.tif'
image_pathCamera = 'img/cameraman.tif'

# Apply transformations to each image
apply_transforms(image_pathLena, 'Lena')
apply_transforms(image_pathHouse, 'House')
apply_transforms(image_pathCamera, 'CameraMan')
- Translação utilizando qualquer parâmetro
def apply_translation(image_path, shift_x, shift_y, title):
    # Carregar a imagem
    image = imageio.imread(image_path)
    
    # Aplicar a translação
    translated_image = ndimage.shift(image, (shift_y, shift_x))
    return translated_image

# Definir o vetor de translação
shift_x = 50  
shift_y = -30

# Caminhos para as imagens
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

# Aplicar a translação a cada imagem
translated_image_lena = apply_translation(lena_path, shift_x, shift_y, 'Lena')
translated_image_cam = apply_translation(cam_path, shift_x, shift_y, 'Cameraman')
translated_image_house = apply_translation(house_path, shift_x, shift_y, 'House')

# Exibir as imagens traduzidas usando matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=50 y=-30")
plt2.title.set_text("Cameraman Translação x=50 y=-30")
plt3.title.set_text("House Translação x=50 y=-30")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
- Translação de 35 no eixo x e 45 no eixo y
def apply_translation(image_path, shift_x, shift_y):
    # Carregar a imagem
    image = imageio.imread(image_path)
    
    # Aplicar a translação
    translated_image = ndimage.shift(image, (shift_y, shift_x))
    return translated_image

# Definir o vetor de translação
shift_x = 35  
shift_y = 45

# Caminhos para as imagens
lena_path = 'img/lena.jpg'
cam_path = 'img/cameraman.tif'
house_path = 'img/house.tif'

# Aplicar a translação a cada imagem
translated_image_lena = apply_translation(lena_path, shift_x, shift_y)
translated_image_cam = apply_translation(cam_path, shift_x, shift_y)
translated_image_house = apply_translation(house_path, shift_x, shift_y)

# Exibir as imagens traduzidas usando matplotlib
fig = plt.figure(figsize=(15, 5))
plt1 = plt.subplot(1, 3, 1)
plt2 = plt.subplot(1, 3, 2)
plt3 = plt.subplot(1, 3, 3)
plt1.title.set_text("Lena Translação x=35 y=45")
plt2.title.set_text("Cameraman Translação x=35 y=45")
plt3.title.set_text("House Translação x=35 y=45")
plt.subplots_adjust(wspace=0.5)

plt1.imshow(translated_image_lena, cmap="gray")
plt2.imshow(translated_image_cam, cmap="gray")
plt3.imshow(translated_image_house, cmap="gray")

plt.show()
## Aula 5 - 28/08 : Transformações de Intensidade

- Aplicar a transformação logarítmica, testar vários valores para o parâmetro c "s = c log (1 + r)"
- Aplicar a transformação de potência (gama), testar vários valores para o parâmetro γ e c=1 "s = crγ"
- Implemente a representação de cada plano de bits das imagens
- Implementar a equalização do histograma 
- Elaborar relatório explicando a implementação de cada transformação e qual foi o efeito na imagem.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def log_transform(c, img):
    return c * np.log(1 + img)

def power_transform(c, gamma, img):
    return c * np.power(img, gamma)

def bit_plane(img, bit):
    return (img & (1 << bit)) >> bit

enhance_path = 'img/enhance-me.gif'
Fig0308_path = 'img/Fig0308(a)(fractured_spine).tif'

# Leitura das imagens
enhance_img = np.array(Image.open(enhance_path))
Fig0308_img = np.array(Image.open(Fig0308_path))

# Aplicação das transformações
c_values = [1, 5, 10, 20]
gamma_values = [0.1, 0.5, 1, 2, 5]

for c in c_values:
    # Transformação Logarítmica
    transformed_enhance = log_transform(c, enhance_img)
    transformed_Fig0308 = log_transform(c, Fig0308_img)
    
    # Exibição
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(transformed_enhance, cmap='gray')
    axs[0].set_title(f'Enhance-me (c = {c})')
    axs[1].imshow(transformed_Fig0308, cmap='gray')
    axs[1].set_title(f'Fig0308 (c = {c})')
    plt.show()

for gamma in gamma_values:
    # Transformação de Potência (Gama)
    transformed_enhance = power_transform(1, gamma, enhance_img)
    transformed_Fig0308 = power_transform(1, gamma, Fig0308_img)
    
    # Exibição
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(transformed_enhance, cmap='gray')
    axs[0].set_title(f'Enhance-me (γ = {gamma})')
    axs[1].imshow(transformed_Fig0308, cmap='gray')
    axs[1].set_title(f'Fig0308 (γ = {gamma})')
    plt.show()

# Representação de cada plano de bits
for i in range(8):
    bit_enhance = bit_plane(enhance_img, i)
    bit_Fig0308 = bit_plane(Fig0308_img, i)
    
    # Exibição
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(bit_enhance, cmap='gray')
    axs[0].set_title(f'Enhance-me (Bit Plane {i})')
    axs[1].imshow(bit_Fig0308, cmap='gray')
    axs[1].set_title(f'Fig0308 (Bit Plane {i})')
    plt.show()

# Equalização do histograma
equalized_enhance = cv2.equalizeHist(enhance_img)
equalized_Fig0308 = cv2.equalizeHist(Fig0308_img)

# Exibição
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(equalized_enhance, cmap='gray')
axs[0].set_title('Enhance-me (Equalização)')
axs[1].imshow(equalized_Fig0308, cmap='gray')
axs[1].set_title('Fig0308 (Equalização)')
plt.show()
## Relatório do Exercício

### 1. Introdução

O processamento digital de imagens (PDI) refere-se à manipulação e análise de imagens utilizando computadores e algoritmos específicos. Este relatório aborda quatro técnicas fundamentais de PDI: transformação logarítmica, transformação de potência (gama), representação de plano de bits e equalização de histograma. Cada técnica é detalhadamente discutida em termos de implementação e impacto visual.

### 2. Transformação Logarítmica

A transformação logarítmica é uma técnica que visa expandir a escala de valores de pixels escuros e comprimir a de pixels claros. É particularmente útil em imagens com grandes variações de brilho.

Utilizando a biblioteca numpy, a transformação é aplicada através da fórmula:

s=c×log(1+r)
onde r representa o valor do pixel original e c é uma constante que controla o contraste da transformação.

Ao variar o valor de c, observa-se uma alteração no contraste da imagem. Valores maiores de c resultam em um contraste mais acentuado, enquanto valores menores suavizam o efeito.

### 3. Transformação Potência (Gama)

A transformação de potência, também conhecida como correção gama, é utilizada para controlar o brilho de uma imagem. Ela é especialmente útil para ajustar imagens que foram capturadas sob diferentes condições de iluminação.

A transformação é dada pela fórmula:
s=c×r^y

onde y é o valor que determina o grau de correção.

### 4. Representação de Cada Plano de Bits

Uma imagem em escala de cinza pode ser representada em diferentes planos de bits, cada um correspondendo a um bit específico no valor do pixel.

A representação de cada plano de bits é obtida isolando-se cada bit do valor do pixel.

O plano de bit mais significativo contém a maior parte da informação visual. Planos de bits inferiores contribuem com detalhes mais sutis e, frequentemente, com ruído.

### 5. Equalização do Histograma

A equalização de histograma é uma técnica que redistribui os valores de pixel de uma imagem para produzir um histograma uniforme.

Utilizando a biblioteca OpenCV, a equalização é realizada através da função cv2.equalizeHist().

A técnica melhora o contraste global da imagem, tornando os detalhes mais visíveis e distribuindo a intensidade dos pixels de forma mais uniforme.
## Aula 6 - 04/09: Filtragem Espacial

- Implementar a operação de convolução.
- Utilizando OPENCV, scipy função convolve e implementação manual.
- Implementar seguintes máscaras:
    - Média
    - Guassiano
    - Laplaciano
    - Sobel X
    - Sobel Y
    - Gradiente (Sobel X + Sobel Y)
    - Laplaciano somado a imagem original
- Utilizar as imagens já disponibilizadas: biel, lena, cameraman, etc.
OPENCV
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolucao_opencv(imagem, kernel):
    return cv2.filter2D(imagem, -1, kernel)

media = np.ones((3, 3)) / 9
gaussiano = cv2.getGaussianKernel(5, 1) * cv2.getGaussianKernel(5, 1).T
laplaciano = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

imagens_info = {
    "lena": {
        "path": "lena_gray_512.tif",
        "data": None
    },
    "biel": {
        "path": "biel.png",
        "data": None
    },
    "cameraman": {
        "path": "cameraman.tif",
        "data": None
    }
}

diretorio = "./img/"

for nome, info in imagens_info.items():
    imagem = cv2.imread(diretorio + info["path"], cv2.IMREAD_GRAYSCALE)
    imagens_info[nome]["data"] = imagem

for nome, info in imagens_info.items():
    imagem = info["data"]

    imagem_media = convolucao_opencv(imagem, media)
    imagem_gauss = convolucao_opencv(imagem, gaussiano)
    imagem_laplac = convolucao_opencv(imagem, laplaciano)
    imagem_sobel_x = convolucao_opencv(imagem, sobel_x)
    imagem_sobel_y = convolucao_opencv(imagem, sobel_y)
    imagem_gradiente = np.sqrt(imagem_sobel_x**2 + imagem_sobel_y**2)
    imagem_laplac_original = imagem + imagem_laplac

    fig, axs = plt.subplots(1, 8, figsize=(25, 5))
    axs[0].imshow(imagem, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(imagem_media, cmap='gray')
    axs[1].set_title('Média')
    axs[1].axis('off')
    axs[2].imshow(imagem_gauss, cmap='gray')
    axs[2].set_title('Gaussiano')
    axs[2].axis('off')
    axs[3].imshow(imagem_laplac, cmap='gray')
    axs[3].set_title('Laplaciano')
    axs[3].axis('off')
    axs[4].imshow(imagem_sobel_x, cmap='gray')
    axs[4].set_title('Sobel X')
    axs[4].axis('off')
    axs[5].imshow(imagem_sobel_y, cmap='gray')
    axs[5].set_title('Sobel Y')
    axs[5].axis('off')
    axs[6].imshow(imagem_gradiente, cmap='gray')
    axs[6].set_title('Gradiente')
    axs[6].axis('off')
    axs[7].imshow(imagem_laplac_original, cmap='gray')
    axs[7].set_title('Laplac + Original')
    axs[7].axis('off')

    plt.tight_layout()
    plt.show()
### Scipy
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def convolucao_scipy(imagem, kernel):
    return convolve2d(imagem, kernel, mode='same', boundary='wrap')

for nome, info in imagens_info.items():
    imagem = cv2.imread(diretorio + info["path"], cv2.IMREAD_GRAYSCALE)
    imagens_info[nome]["data"] = imagem

for nome, info in imagens_info.items():
    imagem = info["data"]

    imagem_media = convolucao_scipy(imagem, media)
    imagem_gauss = convolucao_scipy(imagem, gaussiano)
    imagem_laplac = convolucao_scipy(imagem, laplaciano)
    imagem_sobel_x = convolucao_scipy(imagem, sobel_x)
    imagem_sobel_y = convolucao_scipy(imagem, sobel_y)
    imagem_gradiente = np.sqrt(imagem_sobel_x**2 + imagem_sobel_y**2)
    imagem_laplac_original = imagem + imagem_laplac
    
    fig, axs = plt.subplots(1, 8, figsize=(25, 5))
    axs[0].imshow(imagem, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(imagem_media, cmap='gray')
    axs[1].set_title('Média')
    axs[1].axis('off')
    axs[2].imshow(imagem_gauss, cmap='gray')
    axs[2].set_title('Gaussiano')
    axs[2].axis('off')
    axs[3].imshow(imagem_laplac, cmap='gray')
    axs[3].set_title('Laplaciano')
    axs[3].axis('off')
    axs[4].imshow(imagem_sobel_x, cmap='gray')
    axs[4].set_title('Sobel X')
    axs[4].axis('off')
    axs[5].imshow(imagem_sobel_y, cmap='gray')
    axs[5].set_title('Sobel Y')
    axs[5].axis('off')
    axs[6].imshow(imagem_gradiente, cmap='gray')
    axs[6].set_title('Gradiente')
    axs[6].axis('off')
    axs[7].imshow(imagem_laplac_original, cmap='gray')
    axs[7].set_title('Laplac + Original')
    axs[7].axis('off')

    plt.tight_layout()
    plt.show()
### Método Manual
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolucao_manual(imagem, kernel):
    if len(imagem.shape) == 3:  # Verifica se a imagem é colorida (3 canais)
        altura, largura, canais = imagem.shape
    else:  # Caso contrário, é uma imagem em escala de cinza
        altura, largura = imagem.shape
        canais = 1
        imagem = imagem[:, :, np.newaxis]  # Transforma a imagem em um array 3D para uniformidade no processamento

    k_altura, k_largura = kernel.shape
    padding_altura = k_altura // 2
    padding_largura = k_largura // 2

    imagem_padded = np.pad(imagem, ((padding_altura, padding_altura), (padding_largura, padding_largura), (0, 0)), mode='constant')
    saida = np.zeros_like(imagem)

    for canal in range(canais):
        for y in range(altura):
            for x in range(largura):
                saida[y, x, canal] = np.sum(imagem_padded[y:y + k_altura, x:x + k_largura, canal] * kernel)

    if canais == 1:
        saida = saida[:, :, 0]  # Retorna para a forma 2D se a imagem original estava em escala de cinza

    return saida

for nome, info in imagens_info.items():
    imagem = cv2.imread(diretorio + info["path"], cv2.IMREAD_GRAYSCALE)
    imagens_info[nome]["data"] = imagem

for nome, info in imagens_info.items():
    imagem = info["data"]

    imagem_media = convolucao_opencv(imagem, media)
    imagem_gauss = convolucao_opencv(imagem, gaussiano)
    imagem_laplac = convolucao_opencv(imagem, laplaciano)
    imagem_sobel_x = convolucao_opencv(imagem, sobel_x)
    imagem_sobel_y = convolucao_opencv(imagem, sobel_y)
    imagem_gradiente = np.sqrt(imagem_sobel_x**2 + imagem_sobel_y**2)
    imagem_laplac_original = imagem + imagem_laplac

    fig, axs = plt.subplots(1, 8, figsize=(25, 5))
    axs[0].imshow(imagem, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(imagem_media, cmap='gray')
    axs[1].set_title('Média')
    axs[1].axis('off')
    axs[2].imshow(imagem_gauss, cmap='gray')
    axs[2].set_title('Gaussiano')
    axs[2].axis('off')
    axs[3].imshow(imagem_laplac, cmap='gray')
    axs[3].set_title('Laplaciano')
    axs[3].axis('off')
    axs[4].imshow(imagem_sobel_x, cmap='gray')
    axs[4].set_title('Sobel X')
    axs[4].axis('off')
    axs[5].imshow(imagem_sobel_y, cmap='gray')
    axs[5].set_title('Sobel Y')
    axs[5].axis('off')
    axs[6].imshow(imagem_gradiente, cmap='gray')
    axs[6].set_title('Gradiente')
    axs[6].axis('off')
    axs[7].imshow(imagem_laplac_original, cmap='gray')
    axs[7].set_title('Laplac + Original')
    axs[7].axis('off')

    plt.tight_layout()
    plt.show()
## Aula 7 - 11/09: Transforma de Fourier

- Implementar a Transformada de Fourier (Utilize a biblioteca de sua preferência)
- Plotar o espectro e fase.
- Plotar o espectro 3D (Pesquisar formas de visualização 3D em Python)
    - Utilizar as imagens disponibilizadas na aula (Images_fourier.rar)
    - Criar uma imagem fundo branco e um quadrado simulando a função SINC

### Espectro e Fase de cada imagem:

Car

import numpy as np
import matplotlib.pyplot as plt

# Carregue a imagem de sua escolha (por exemplo, 'Imgcar')
imgCar = plt.imread("./img/car.tif")

# Função para calcular a Transformada de Fourier 2D e plotar a imagem original, o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote a imagem original, o espectro de magnitude e a fase
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro de Magnitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(imgCar, "imgCar")
len_periodic_noise
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carregue a imagem de sua escolha
imgLen = cv2.imread("./img/len_periodic_noise.png", cv2.IMREAD_GRAYSCALE)

# Função para calcular a Transformada de Fourier 2D e plotar a imagem original, o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote a imagem original, o espectro de magnitude e a fase
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro de Magnitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(imgLen, "imgLen")


newspaper_shot_woman
import numpy as np
import matplotlib.pyplot as plt

# Carregue a imagem de sua escolha (por exemplo, 'Imgcar')
imgNS = plt.imread("./img/newspaper_shot_woman.tif")

# Função para calcular a Transformada de Fourier 2D e plotar a imagem original, o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote a imagem original, o espectro de magnitude e a fase
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro de Magnitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(imgNS, "imgNS")
periodic_noise
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carregue a imagem de sua escolha
imgPeriodic = cv2.imread("./img/periodic_noise.png", cv2.IMREAD_GRAYSCALE)

# Função para calcular a Transformada de Fourier 2D e plotar a imagem original, o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote a imagem original, o espectro de magnitude e a fase
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro de Magnitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(imgPeriodic, "imgPeriodic")

sinc
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carregue a imagem de sua escolha
imgSinc = cv2.imread("./img/sinc.png", cv2.IMREAD_GRAYSCALE)

# Função para calcular a Transformada de Fourier 2D e plotar a imagem original, o espectro e a fase
def calcular_e_plotar_fft(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(transformada_fourier)

    # Calcule a fase
    fase = np.angle(transformada_fourier)

    # Plote a imagem original, o espectro de magnitude e a fase
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + espectro_magnitude), cmap='gray')
    plt.title('Espectro de Magnitude')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(fase, cmap='gray')
    plt.title('Fase')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Chamada da função para cada imagem
calcular_e_plotar_fft(imgSinc, "imgSinc")

### Espectro 3D 

car
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D')
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carregue a imagem de sua escolha (por exemplo, 'imgCar')
imgCar = cv2.imread("./img/car.tif", cv2.IMREAD_GRAYSCALE)


# Chamada da função para cada imagem
plotar_espectro_3D(imgCar, "imgCar")


len_periodic_noise
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D')
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carregue a imagem de sua escolha (por exemplo, 'imgCar')
imgLen = cv2.imread("./img/len_periodic_noise.png", cv2.IMREAD_GRAYSCALE)

# Chamada da função para cada imagem
plotar_espectro_3D(imgLen, "imgLen")


newspaper_shot_woman
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D')
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carregue a imagem de sua escolha (por exemplo, 'imgCar')
imgNS = cv2.imread("./img/newspaper_shot_woman.tif", cv2.IMREAD_GRAYSCALE)

# Chamada da função para cada imagem
plotar_espectro_3D(imgNS, "imgNS")


periodic_noise
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D')
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carregue a imagem de sua escolha (por exemplo, 'imgCar')
imgPeriodic = cv2.imread("./img/periodic_noise.png", cv2.IMREAD_GRAYSCALE)

# Chamada da função para cada imagem
plotar_espectro_3D(imgPeriodic, "imgPeriodic")

sinc
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Função para calcular e plotar o espectro 3D
def plotar_espectro_3D(imagem, titulo):
    # Calcule a Transformada de Fourier 2D da imagem
    transformada_fourier = np.fft.fft2(imagem)

    # Calcule o espectro 2D
    espectro_2D = np.fft.fftshift(transformada_fourier)

    # Calcule o espectro de magnitude
    espectro_magnitude = np.abs(espectro_2D)

    # Crie uma grade de coordenadas para o espectro 3D
    x = np.fft.fftshift(np.fft.fftfreq(imagem.shape[1]))
    y = np.fft.fftshift(np.fft.fftfreq(imagem.shape[0]))
    X, Y = np.meshgrid(x, y)

    # Plote o espectro 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Espectro 3D')
    ax.plot_surface(X, Y, np.log(1 + espectro_magnitude), cmap='viridis')

    plt.show()

# Carregue a imagem de sua escolha (por exemplo, 'imgCar')
imgSinc = cv2.imread("./img/sinc.png", cv2.IMREAD_GRAYSCALE)

# Chamada da função para cada imagem
plotar_espectro_3D(imgSinc, "imgSinc")

## Aula 7.2 - 12/09: Combining Images

Objetivo: Implementar códigos que utilizam operações básicas combinando duas imagens.

- Verificação de defeitos em placas: Basicamente realizando uma operação de subtração entre uma imagem de uma placa sem defeito com uma placa com defeito é possivel encontrar defeitos no processo de fabricação: 
    - https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/3-Combining-Images/Defect_Detection/

- Detecção de movimento: A partir de um vídeo, ao realizar a subtração do fundo da cena sem nenhuma pessoa é possível detectar movimentos: 
    - https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/3-Combining-Images/Background_Subtraction/
placa
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

placa_ori = np.array(Image.open(r'./img/imgCropped.png'))
placa_def = np.array(Image.open(r'./img/imgDefeito.png'))

row = placa_ori.shape[1]
col = placa_ori.shape[0]
xShift = 10
yShift = 10
registImg = np.zeros(placa_ori.shape)
registImg[yShift + 1 : row, xShift + 1 : col] = placa_def[1 : row - yShift, 1 : col - xShift]

fig = plt.figure(figsize=(10, 5))
plt1 = plt.subplot(1, 4, 1)
plt2 = plt.subplot(1, 4, 2)
plt3 = plt.subplot(1, 4, 3)
plt4 = plt.subplot(1, 4, 4)
plt1.title.set_text("Original")
plt2.title.set_text('Defeito')
plt3.title.set_text('Sem translação')
plt4.title.set_text('Com translação')
plt1.imshow(placa_ori, cmap='gray')
plt2.imshow(placa_def, cmap='gray')
plt3.imshow((placa_ori - placa_def), cmap='gray')
plt4.imshow(placa_ori - registImg, cmap='gray')
plt.subplots_adjust(wspace=0.5)
video
import cv2
import numpy as np

# Abrir o vídeo
video_capture = cv2.VideoCapture('./video/surveillance.mpg')

# Configurar a gravação do vídeo de saída
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter('Background_Subtraction.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# Parâmetros para a subtração de fundo
alpha = 0.95
theta = 0.1
background = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if background is None:
        background = gray_frame.astype(float)
        continue

    # Atualizar o modelo de fundo com suavização exponencial
    background = alpha * background + (1 - alpha) * gray_frame

    # Calcular a diferença entre o frame atual e o fundo
    diff_frame = np.abs(gray_frame - background)
    thresh_frame = (diff_frame > theta * 255).astype(np.uint8)

    # Exibir as imagens
    cv2.imshow('New frame', gray_frame)
    cv2.imshow('Background frame', background.astype(np.uint8))
    cv2.imshow('Difference image', diff_frame.astype(np.uint8))
    cv2.imshow('Thresholded difference image', thresh_frame * 255)

    # Gravar o frame de saída
    out.write(cv2.cvtColor(thresh_frame * 255, cv2.COLOR_GRAY2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
out.release()
cv2.destroyAllWindows()
## Aula 8 - 18/09: Filtragem Frequência

1.     Calcule e visualize o espectro de uma imagem 512x512 pixels:

    a)  crie e visualize uma imagem simples – quadrado branco sobre fundo preto;

    b)  calcular e visualizar seu espectro de Fourier (amplitudes);

    c)  calcular e visualizar seu espectro de Fourier (fases);

    d)  obter e visualizar seu espectro de Fourier centralizado;

    e)  Aplique uma rotação de 40º no quadrado e repita os passo b-d;

    f)  Aplique uma translação nos eixos x e y no quadrado e repita os passo b-d;

    g)  Aplique um zoom na imagem e repita os passo b-d;

    h)  Explique o que acontece com a transformada de Fourier quando é aplicado a rotação, translação e zoom.


2.     Crie filtros passa-baixa do tipo ideal, butterworth e gaussiano e aplique-o às imagens disponibilizadas. Visualize o seguinte:

    a)  a imagem inicial;

    b)  a imagem de cada filtro;

    c)  a imagem resultante após aplicação de cada filtro.
 

3.     Crie um filtro passa-alta do tipo ideal, butterworth e gaussiano e aplique-o às imagens disponibilizadas. Visualize os mesmos dados da tarefa anterior:

    a)  a imagem inicial;

    b)  a imagem de cada filtro;

    c)  a imagem resultante após aplicação de cada filtro.


4.     Varie o parâmetro de frequência de corte no filtro passa-baixa criado na tarefa 2. Por exemplo, tome valores de D0 iguais a 0,01, 0,05, 0,5. A imagem inicial é igual à anterior. Visualize as imagens dos filtros e as imagens resultantes. Explique os resultados.

5.     Efetue o mesmo que se pede no item 4, mas use o filtro passa-alta em vez do filtro passa-baixa.

6.      Além dos filtros passa-baixa e passa-alta também existe o filtro passa-banda? Explique seu funcionamento e aplique um filtro passa-banda na imagem.
1.      Calcule e visualize o espectro de uma imagem 512x512 pixels:   
    a)  crie e visualize uma imagem simples – quadrado branco sobre fundo preto;
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Criando uma imagem preta de 512x512
imagem = np.zeros((512, 512), dtype=np.uint8)

# Adicionando um quadrado branco no meio da imagem
cv2.rectangle(imagem, (204, 204), (308, 308), 255, -1)

# Visualizando a imagem
plt.imshow(imagem, cmap='gray')
plt.title("Imagem Original")
plt.show()
     b)  calcular e visualizar seu espectro de Fourier (amplitudes);
# Calculando a Transformada de Fourier
f = np.fft.fft2(imagem)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Visualização
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro de Fourier - Amplitudes')
plt.show()
    c)  calcular e visualizar seu espectro de Fourier (fases);
# Calculando as fases
fase_spectrum = np.angle(fshift)

# Visualização
plt.imshow(fase_spectrum, cmap = 'gray')
plt.title('Espectro de Fourier - Fases')
plt.show()
    d)  obter e visualizar seu espectro de Fourier centralizado
# Calculando a Transformada de Fourier
f = np.fft.fft2(imagem)
fshift = np.fft.fftshift(f)  # Centralizando o espectro
magnitude_spectrum_centered = 20 * np.log(np.abs(f) + 1)  # Adicionamos 1 para evitar log(0)

# Visualização do espectro centralizado
plt.imshow(magnitude_spectrum_centered, cmap='gray')
plt.title('Espectro de Fourier Centralizado')
plt.show()
    e)  Aplique uma rotação de 40º no quadrado e repita os passo b-d;
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Código de geração da imagem do quadrado vai aqui...

# Aplicando rotação de 40º na imagem
rows, cols = imagem.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 40, 1)
imagem_rotacionada = cv2.warpAffine(imagem, M, (cols, rows))

# b) Calcular e visualizar o espectro de Fourier (amplitudes) da imagem rotacionada
f_rot = np.fft.fft2(imagem_rotacionada)
fshift_rot = np.fft.fftshift(f_rot)
magnitude_spectrum_rot = 20*np.log(np.abs(fshift_rot))

# c) Calcular e visualizar o espectro de Fourier (fases) da imagem rotacionada
fase_spectrum_rot = np.angle(fshift_rot)

# Utilizando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem rotacionada
axs[0].imshow(imagem_rotacionada, cmap='gray')
axs[0].set_title('Imagem Rotacionada')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_rot, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_rot, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')

plt.tight_layout()
plt.show()
    f)  Aplique uma translação nos eixos x e y no quadrado e repita os passo b-d;
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Criação da imagem original: um quadrado branco em um fundo preto
imagem = np.zeros((512, 512), dtype=np.uint8)
cv2.rectangle(imagem, (154, 154), (358, 358), 255, -1)

# Aplicando translação de 40 pixels nos eixos x e y na imagem original
translacao = np.float32([[1, 0, 40], [0, 1, 40]])
imagem_transladada = cv2.warpAffine(imagem, translacao, (512, 512))

# b) Calculando o espectro de Fourier (amplitudes) da imagem transladada
f_trans = np.fft.fft2(imagem_transladada)
fshift_trans = np.fft.fftshift(f_trans)
magnitude_spectrum_trans = 20*np.log(np.abs(fshift_trans) + 1)  # +1 para evitar log(0)

# c) Calculando o espectro de Fourier (fases) da imagem transladada
fase_spectrum_trans = np.angle(fshift_trans)

# Usando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem transladada
axs[0].imshow(imagem_transladada, cmap='gray')
axs[0].set_title('Imagem Transladada')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_trans, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_trans, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')

plt.tight_layout()
plt.show()
    g)  Aplique um zoom na imagem e repita os passo b-d;
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Criação da imagem original: um quadrado branco em um fundo preto
imagem = np.zeros((512, 512), dtype=np.uint8)
cv2.rectangle(imagem, (154, 154), (358, 358), 255, -1)

# Aplicando zoom: reduzindo a imagem em 50% e depois aumentando para o tamanho original
imagem_zoom = cv2.resize(imagem, (256, 256))
imagem_zoom = cv2.resize(imagem_zoom, (512, 512))

# b) Calculando o espectro de Fourier (amplitudes) da imagem com zoom
f_zoom = np.fft.fft2(imagem_zoom)
fshift_zoom = np.fft.fftshift(f_zoom)
magnitude_spectrum_zoom = 20*np.log(np.abs(fshift_zoom) + 1)  # +1 para evitar log(0)

# c) Calculando o espectro de Fourier (fases) da imagem com zoom
fase_spectrum_zoom = np.angle(fshift_zoom)

# Usando subplots para exibir as imagens lado a lado
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Imagem com zoom
axs[0].imshow(imagem_zoom, cmap='gray')
axs[0].set_title('Imagem com Zoom')
axs[0].axis('off')

# Espectro de Amplitude
axs[1].imshow(magnitude_spectrum_zoom, cmap='gray')
axs[1].set_title('Espectro de Fourier - Amplitude')
axs[1].axis('off')

# Espectro de Fase
axs[2].imshow(fase_spectrum_zoom, cmap='gray')
axs[2].set_title('Espectro de Fourier - Fase')
axs[2].axis('off')

plt.tight_layout()
plt.show()
    h)  Explique o que acontece com a transformada de Fourier quando é aplicado a rotação, translação e zoom.
Rotação: O espectro de Fourier também é rotacionado. A rotação da imagem no espaço espacial resulta em uma rotação correspondente no espectro de Fourier.

Translação: A translação da imagem resulta em uma modulação do espectro de Fourier. A amplitude não muda, mas a fase é modulada.

Zoom: O zoom (ampliação ou redução) da imagem altera a densidade de energia no espectro de Fourier. Uma ampliação causa uma condensação do espectro e uma redução causa uma expansão.
2.     Crie filtros passa-baixa do tipo ideal, butterworth e gaussiano e aplique-o às imagens disponibilizadas. Visualize o seguinte:

    a)  a imagem inicial;
imgOrg = './img/sinc_original.png'
imgMenor = './img/sinc_original_menor.tif'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue todas as imagens
imgOrg = cv2.imread('./img/sinc_original.png', cv2.IMREAD_GRAYSCALE)
imgRot = cv2.imread('./img/sinc_rot.png', cv2.IMREAD_GRAYSCALE)
imgRot2 = cv2.imread('./img/sinc_rot2.png', cv2.IMREAD_GRAYSCALE)
imgTrans = cv2.imread('./img/sinc_trans.png', cv2.IMREAD_GRAYSCALE)
imgZoom = cv2.imread('./img/sinc_zoom.png', cv2.IMREAD_GRAYSCALE)

# Crie uma figura para organizar as imagens e legendas
plt.figure(figsize=(15, 5))

# Imagem Original
plt.subplot(151)
plt.imshow(imgOrg, cmap='gray')
plt.title('a) Imagem Original')
plt.axis('off')

# Imagem Rotacionada (40º)
plt.subplot(152)
plt.imshow(imgRot, cmap='gray')
plt.title('e) Imagem Rotacionada (40º)')
plt.axis('off')

# Imagem Rotacionada (20º)
plt.subplot(153)
plt.imshow(imgRot2, cmap='gray')
plt.title('e) Imagem Rotacionada (20º)')
plt.axis('off')

# Imagem Transladada
plt.subplot(154)
plt.imshow(imgTrans, cmap='gray')
plt.title('f) Imagem Transladada')
plt.axis('off')

plt.tight_layout()
plt.show()

    b)  a imagem de cada filtro;

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar um filtro passa-baixa do tipo ideal
def ideal_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= D0:
                filter[i, j] = 1
    
    return filter

# Função para criar um filtro passa-baixa de Butterworth
def butterworth_lowpass_filter(shape, D0, n):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 / (1 + (distance / D0)**(2 * n))
    
    return filter

# Função para criar um filtro passa-baixa Gaussiano
def gaussian_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = np.exp(-distance**2 / (2 * D0**2))
    
    return filter

# Caminhos das imagens
imgOrg = './img/sinc_original.png'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'

# Lista de caminhos das imagens
image_paths = [imgOrg, imgRot, imgRot2, imgTrans]

# Tamanho do filtro passa-baixa (ajuste conforme necessário)
D0 = 50

# Lista de nomes dos filtros
filter_names = ['Ideal', 'Butterworth', 'Gaussiano']

# Criar figuras para cada imagem
plt.figure(figsize=(15, 15))

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros
    ideal_filter = ideal_lowpass_filter(img.shape, D0)
    butterworth_filter = butterworth_lowpass_filter(img.shape, D0, n=2)  # O valor de "n" pode ser ajustado
    gaussian_filter = gaussian_lowpass_filter(img.shape, D0)
    
    filtered_spectrum_ideal = fourier_spectrum * ideal_filter
    filtered_spectrum_butterworth = fourier_spectrum * butterworth_filter
    filtered_spectrum_gaussian = fourier_spectrum * gaussian_filter
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal)))
    image_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth)))
    image_filtered_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian)))
    
    # Plotar a imagem original e os espectros de Fourier após a aplicação dos filtros
    plt.subplot(5, 4, i * 4 + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagem {i+1}')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 2)
    plt.imshow(np.log(np.abs(fourier_spectrum) + 1), cmap='gray')
    plt.title('Espectro Original')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 3)
    plt.imshow(np.log(np.abs(filtered_spectrum_ideal) + 1), cmap='gray')
    plt.title(f'Espectro Ideal ({filter_names[0]})')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 4)
    plt.imshow(np.log(np.abs(filtered_spectrum_butterworth) + 1), cmap='gray')
    plt.title(f'Espectro Butterworth ({filter_names[1]})')
    plt.axis('off')

# Ajustar o layout
plt.tight_layout()
plt.show()

    c) a imagem de cada filtro; 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar um filtro passa-baixa do tipo ideal
def ideal_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= D0:
                filter[i, j] = 1
    
    return filter

# Função para criar um filtro passa-baixa de Butterworth
def butterworth_lowpass_filter(shape, D0, n):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 / (1 + (distance / D0)**(2 * n))
    
    return filter

# Função para criar um filtro passa-baixa Gaussiano
def gaussian_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = np.exp(-distance**2 / (2 * D0**2))
    
    return filter

# Caminhos das imagens
imgOrg = './img/sinc_original.png'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'

# Lista de caminhos das imagens
image_paths = [imgOrg, imgRot, imgRot2, imgTrans]

# Tamanho do filtro passa-baixa (ajuste conforme necessário)
D0 = 50

# Lista de nomes dos filtros
filter_names = ['Ideal', 'Butterworth', 'Gaussiano']

# Criar figuras para cada imagem
plt.figure(figsize=(15, 15))

# Lista para armazenar as imagens filtradas
filtered_images = []

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        continue
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros
    ideal_filter = ideal_lowpass_filter(img.shape, D0)
    butterworth_filter = butterworth_lowpass_filter(img.shape, D0, n=2)  # O valor de "n" pode ser ajustado
    gaussian_filter = gaussian_lowpass_filter(img.shape, D0)
    
    filtered_spectrum_ideal = fourier_spectrum * ideal_filter
    filtered_spectrum_butterworth = fourier_spectrum * butterworth_filter
    filtered_spectrum_gaussian = fourier_spectrum * gaussian_filter
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal)))
    image_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth)))
    image_filtered_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian)))
    
    filtered_images.append([image_filtered_ideal, image_filtered_butterworth, image_filtered_gaussian])

# Exibir as imagens após a aplicação de cada filtro
plt.figure(figsize=(15, 5))

for i, image_set in enumerate(filtered_images):
    for j, filtered_image in enumerate(image_set):
        plt.subplot(len(image_paths), 3, i * 3 + j + 1)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtro {filter_names[j]} ({chr(ord("a") + i)})')
        plt.axis('off')

plt.tight_layout()
plt.show()

    d) a imagem resultante após aplicação de cada filtro.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar um filtro passa-baixa do tipo ideal
def ideal_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= D0:
                filter[i, j] = 1
    
    return filter

# Função para criar um filtro passa-baixa de Butterworth
def butterworth_lowpass_filter(shape, D0, n):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 / (1 + (distance / D0)**(2 * n))
    
    return filter

# Função para criar um filtro passa-baixa Gaussiano
def gaussian_lowpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = np.exp(-distance**2 / (2 * D0**2))
    
    return filter

# Caminhos das imagens
imgOrg = './img/sinc_original.png'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'

# Lista de caminhos das imagens
image_paths = [imgOrg, imgRot, imgRot2, imgTrans]

# Tamanho do filtro passa-baixa (ajuste conforme necessário)
D0 = 50

# Lista de nomes dos filtros
filter_names = ['Ideal', 'Butterworth', 'Gaussiano']

# Criar figuras para cada imagem
plt.figure(figsize=(15, 15))

# Lista para armazenar as imagens filtradas
filtered_images = []

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        continue
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros
    ideal_filter = ideal_lowpass_filter(img.shape, D0)
    butterworth_filter = butterworth_lowpass_filter(img.shape, D0, n=2)  # O valor de "n" pode ser ajustado
    gaussian_filter = gaussian_lowpass_filter(img.shape, D0)
    
    filtered_spectrum_ideal = fourier_spectrum * ideal_filter
    filtered_spectrum_butterworth = fourier_spectrum * butterworth_filter
    filtered_spectrum_gaussian = fourier_spectrum * gaussian_filter
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal)))
    image_filtered_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth)))
    image_filtered_gaussian = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian)))
    
    filtered_images.append([image_filtered_ideal, image_filtered_butterworth, image_filtered_gaussian])

# Exibir as imagens resultantes após a aplicação de cada filtro
plt.figure(figsize=(15, 5))

for i, image_set in enumerate(filtered_images):
    for j, filtered_image in enumerate(image_set):
        plt.subplot(len(image_paths), 3, i * 3 + j + 1)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtro {filter_names[j]} ({chr(ord("a") + i)})')
        plt.axis('off')

plt.tight_layout()
plt.show()

3.     Crie filtros passa-alta do tipo ideal, butterworth e gaussiano e aplique-o às imagens disponibilizadas. Visualize o seguinte:

    a)  a imagem inicial;
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregue todas as imagens
imgOrg = cv2.imread('./img/sinc_original.png', cv2.IMREAD_GRAYSCALE)
imgRot = cv2.imread('./img/sinc_rot.png', cv2.IMREAD_GRAYSCALE)
imgRot2 = cv2.imread('./img/sinc_rot2.png', cv2.IMREAD_GRAYSCALE)
imgTrans = cv2.imread('./img/sinc_trans.png', cv2.IMREAD_GRAYSCALE)

# Crie uma figura para organizar as imagens e legendas
plt.figure(figsize=(15, 5))

# Imagem Original
plt.subplot(151)
plt.imshow(imgOrg, cmap='gray')
plt.title('a) Imagem Original')
plt.axis('off')

# Imagem Rotacionada (40º)
plt.subplot(152)
plt.imshow(imgRot, cmap='gray')
plt.title('e) Imagem Rotacionada (40º)')
plt.axis('off')

# Imagem Rotacionada (20º)
plt.subplot(153)
plt.imshow(imgRot2, cmap='gray')
plt.title('e) Imagem Rotacionada (20º)')
plt.axis('off')

# Imagem Transladada
plt.subplot(154)
plt.imshow(imgTrans, cmap='gray')
plt.title('f) Imagem Transladada')
plt.axis('off')

plt.tight_layout()
plt.show()

    b) a imagem do spectro de fourier;
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar um filtro passa-alta do tipo ideal
def ideal_highpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= D0:
                filter[i, j] = 0
    
    return filter

# Função para criar um filtro passa-alta de Butterworth
def butterworth_highpass_filter(shape, D0, n):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 / (1 + (D0 / distance)**(2 * n))
    
    return filter

# Função para criar um filtro passa-alta Gaussiano
def gaussian_highpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 - np.exp(-distance**2 / (2 * D0**2))
    
    return filter

# Caminhos das imagens
imgOrg = './img/sinc_original.png'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'

# Lista de caminhos das imagens
image_paths = [imgOrg, imgRot, imgRot2, imgTrans]

# Tamanho do filtro passa-alta (ajuste conforme necessário)
D0_highpass = 50

# Lista de nomes dos filtros
filter_names = ['Ideal', 'Butterworth', 'Gaussiano']

# Criar figuras para cada imagem
plt.figure(figsize=(15, 15))

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        continue
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros passa-alta
    ideal_highpass = ideal_highpass_filter(img.shape, D0_highpass)
    butterworth_highpass = butterworth_highpass_filter(img.shape, D0_highpass, n=2)  # O valor de "n" pode ser ajustado
    gaussian_highpass = gaussian_highpass_filter(img.shape, D0_highpass)
    
    filtered_spectrum_ideal_highpass = fourier_spectrum * ideal_highpass
    filtered_spectrum_butterworth_highpass = fourier_spectrum * butterworth_highpass
    filtered_spectrum_gaussian_highpass = fourier_spectrum * gaussian_highpass
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal_highpass)))
    image_filtered_butterworth_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth_highpass)))
    image_filtered_gaussian_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian_highpass)))
    
    # Plotar a imagem original e os espectros de Fourier após a aplicação dos filtros
    plt.subplot(5, 4, i * 4 + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagem {i+1}')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 2)
    plt.imshow(np.log(np.abs(fourier_spectrum) + 1), cmap='gray')
    plt.title('Espectro Original')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 3)
    plt.imshow(np.log(np.abs(filtered_spectrum_ideal_highpass) + 1), cmap='gray')
    plt.title(f'Espectro Ideal Highpass ({filter_names[0]})')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 4)
    plt.imshow(np.log(np.abs(filtered_spectrum_butterworth_highpass) + 1), cmap='gray')
    plt.title(f'Espectro Butterworth Highpass ({filter_names[1]})')
    plt.axis('off')
    
# Ajustar o layout
plt.tight_layout()
plt.show()

    c) a imagem de cada filtro
# Exibir as imagens resultantes após a aplicação de cada filtro passa-alta
plt.figure(figsize=(15, 15))

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        continue
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros passa-alta
    ideal_highpass = ideal_highpass_filter(img.shape, D0_highpass)
    butterworth_highpass = butterworth_highpass_filter(img.shape, D0_highpass, n=2)  # O valor de "n" pode ser ajustado
    gaussian_highpass = gaussian_highpass_filter(img.shape, D0_highpass)
    
    filtered_spectrum_ideal_highpass = fourier_spectrum * ideal_highpass
    filtered_spectrum_butterworth_highpass = fourier_spectrum * butterworth_highpass
    filtered_spectrum_gaussian_highpass = fourier_spectrum * gaussian_highpass
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal_highpass)))
    image_filtered_butterworth_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth_highpass)))
    image_filtered_gaussian_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian_highpass)))
    
    # Plotar as imagens resultantes após a aplicação de cada filtro passa-alta
    plt.subplot(5, 3, i * 3 + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagem {i+1}')
    plt.axis('off')
    
    plt.subplot(5, 3, i * 3 + 2)
    plt.imshow(image_filtered_ideal_highpass, cmap='gray')
    plt.title(f'Filtro Ideal Highpass')
    plt.axis('off')
    
    plt.subplot(5, 3, i * 3 + 3)
    plt.imshow(image_filtered_butterworth_highpass, cmap='gray')
    plt.title(f'Filtro Butterworth Highpass')
    plt.axis('off')

# Ajustar o layout
plt.tight_layout()
plt.show()

    d) a imagem resultante após aplicação de cada filtro.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para criar um filtro passa-alta do tipo ideal
def ideal_highpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= D0:
                filter[i, j] = 0
    
    return filter

# Função para criar um filtro passa-alta de Butterworth
def butterworth_highpass_filter(shape, D0, n):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 / (1 + (D0 / distance)**(2 * n))
    
    return filter

# Função para criar um filtro passa-alta Gaussiano
def gaussian_highpass_filter(shape, D0):
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            filter[i, j] = 1 - np.exp(-distance**2 / (2 * D0**2))
    
    return filter

# Caminhos das imagens
imgOrg = './img/sinc_original.png'
imgRot = './img/sinc_rot.png'
imgRot2 = './img/sinc_rot2.png'
imgTrans = './img/sinc_trans.png'

# Lista de caminhos das imagens
image_paths = [imgOrg, imgRot, imgRot2, imgTrans]

# Tamanho do filtro passa-alta (ajuste conforme necessário)
D0_highpass = 50

# Lista de nomes dos filtros
filter_names = ['Ideal', 'Butterworth', 'Gaussiano']

# Criar figuras para cada imagem
plt.figure(figsize=(15, 15))

for i, image_path in enumerate(image_paths):
    # Carregar a imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Falha ao carregar a imagem: {image_path}")
        continue
    
    # Transformada de Fourier 2D da imagem
    fourier_spectrum = np.fft.fftshift(np.fft.fft2(img))
    
    # Aplicar os filtros passa-alta
    ideal_highpass = ideal_highpass_filter(img.shape, D0_highpass)
    butterworth_highpass = butterworth_highpass_filter(img.shape, D0_highpass, n=2)  # O valor de "n" pode ser ajustado
    gaussian_highpass = gaussian_highpass_filter(img.shape, D0_highpass)
    
    filtered_spectrum_ideal_highpass = fourier_spectrum * ideal_highpass
    filtered_spectrum_butterworth_highpass = fourier_spectrum * butterworth_highpass
    filtered_spectrum_gaussian_highpass = fourier_spectrum * gaussian_highpass
    
    # Transformada inversa de Fourier para obter as imagens filtradas
    image_filtered_ideal_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_ideal_highpass)))
    image_filtered_butterworth_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_butterworth_highpass)))
    image_filtered_gaussian_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum_gaussian_highpass)))
    
    # Plotar as imagens resultantes após a aplicação de cada filtro passa-alta
    plt.subplot(5, 4, i * 4 + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagem {i+1}')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 2)
    plt.imshow(image_filtered_ideal_highpass, cmap='gray')
    plt.title(f'Filtro Ideal Highpass')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 3)
    plt.imshow(image_filtered_butterworth_highpass, cmap='gray')
    plt.title(f'Filtro Butterworth Highpass')
    plt.axis('off')
    
    plt.subplot(5, 4, i * 4 + 4)
    plt.imshow(image_filtered_gaussian_highpass, cmap='gray')
    plt.title(f'Filtro Gaussiano Highpass')
    plt.axis('off')

# Ajustar o layout
plt.tight_layout()
plt.show()

4.      Varie o parâmetro de frequência de corte no filtro passa-baixa criado na tarefa 2. Por exemplo, tome valores de D0 iguais a 0,01, 0,05, 0,5. A imagem inicial é igual à anterior. Visualize as imagens dos filtros e as imagens resultantes. Explique os resultados.
import cv2
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './img/sinc_original.png'
sinc_original_menor_path = './img/sinc_original_menor.tif'
sinc_rot_path = './img/sinc_rot.png'
sinc_rot2_path = './img/sinc_rot2.png'
sinc_trans_path = './img/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path,
cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)

def fourier_spectrum(image):
# Computa a transformada de Fourier 2D
    f = np.fft.fft2(image)
    # Centraliza as frequências baixas
    fshift = np.fft.fftshift(f)
    # Calcula a magnitude e aplica o logaritmo para melhor visualização
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def ideal_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < cutoff:
                filter[y, x] = 1
    return filter

def butterworth_lowpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = 1 / (1 + (distance((y, x), center) / cutoff)** (2 * order))
        return filter
    
def gaussian_lowpass_filter(image, cutoff):
    rows, cols = image.shape
    center = (rows / 2, cols / 2)
    filter = np.zeros((rows, cols))
    for x in range(cols):
        for y in range(rows):
            filter[y, x] = np.exp(-(distance((y, x), center) ** 2) / (2* (cutoff ** 2)))
    return filter
cutoff = 30

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def apply_filter(image, filter):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def ideal_highpass_filter(image, cutoff):
    return 1 - ideal_lowpass_filter(image, cutoff)

def butterworth_highpass_filter(image, cutoff, order=2):
    return 1 - butterworth_lowpass_filter(image, cutoff, order)

def gaussian_highpass_filter(image, cutoff):
    return 1 - gaussian_lowpass_filter(image, cutoff)

# Função para aplicar o filtro usando transformada de Fourier
def apply_filter(image, filter):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)',
'Transladada']

# Definindo os valores de D0 para variação
cutoffs = [0.01, 0.05, 0.5]

for cutoff in cutoffs:
    for idx, image in enumerate(images):
        # Fourier
        spectrum = fourier_spectrum(image)
        
        # Criação dos filtros
        ideal_lp = ideal_lowpass_filter(image, cutoff)
        butter_lp = butterworth_lowpass_filter(image, cutoff)
        gaussian_lp = gaussian_lowpass_filter(image, cutoff)
        
        # Aplicação dos filtros
        result_ideal = apply_filter(image, ideal_lp)
        result_butter = apply_filter(image, butter_lp)
        result_gaussian = apply_filter(image, gaussian_lp)
        
        # Exibição
        plt.figure(figsize=(20, 10))
        
        # Imagem original
        plt.subplot(4, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'a) Imagem {titles[idx]}')
        plt.axis('off')
    
        # Espectro de Fourier
        plt.subplot(4, 4, 2)
        plt.imshow(spectrum, cmap='gray')
        plt.title('b) Espectro de Fourier')
        plt.axis('off')

        # Filtro passa-baixa Ideal
        plt.subplot(4, 4, 3)
        plt.imshow(ideal_lp, cmap='gray')
        plt.title(f'c) Filtro Ideal Passa-Baixa (D0={cutoff})')
        plt.axis('off')
        
        # Resultado filtro Ideal
        plt.subplot(4, 4, 4)
        plt.imshow(result_ideal, cmap='gray')
        plt.title('d) Após Filtro Ideal')
        plt.axis('off')
        
        # Filtro passa-baixa Butterworth
        plt.subplot(4, 4, 7)
        plt.imshow(butter_lp, cmap='gray')
        plt.title(f'c) Filtro Butterworth Passa-Baixa (D0={cutoff})')
        plt.axis('off')
        
        # Resultado filtro Butterworth
        plt.subplot(4, 4, 8)
        plt.imshow(result_butter, cmap='gray')
        plt.title('d) Após Filtro Butterworth')
        plt.axis('off')
        
        # Filtro passa-baixa Gaussiano
        plt.subplot(4, 4, 11)
        plt.imshow(gaussian_lp, cmap='gray')
        plt.subplot(4, 4, 11)
        plt.imshow(gaussian_lp, cmap='gray')
        plt.title(f'c) Filtro Gaussiano Passa-Baixa (D0={cutoff})')
        plt.axis('off')
        
        # Resultado filtro Gaussiano
        plt.subplot(4, 4, 12)
        plt.imshow(result_gaussian, cmap='gray')
        plt.title('d) Após Filtro Gaussiano')
        plt.axis('off')
        
        # Ajusta o layout e mostra a figura
        plt.tight_layout()
        plt.show()
D0 = 0,01 (Frequência de Corte Extremamente Baixa):
Ao definirmos D0 como 0,01, estamos estabelecendo uma frequência de corte
extremamente baixa. Isso significa que esperamos reter apenas as componentes de
frequência muito baixa da imagem, essencialmente a componente de baixa
frequência DC. Como resultado, a imagem que tivermos será predominantemente
uma versão altamente suavizada, às vezes referida como "borrada", da imagem
original.

D0 = 0,05 (Frequência de Corte Moderadamente Baixa):
Com um valor de D0 igual a 0,05, estamos definindo uma frequência de corte um
pouco mais alta. Isso implica que mais componentes de frequência da imagem serão
preservados em comparação com D0 = 0,01. A imagem resultante ainda
apresentará suavização, mas começarão a surgir detalhes mais sutis em
comparação com a configuração anterior (D0 = 0,01).

D0 = 0,5 (Frequência de Corte Moderada):
Aqui, ao definirmos D0 como 0,5, estamos mantendo a maioria das componentes de
frequência baixa da imagem. Como resultado, a imagem obtida será mais próxima
da imagem original em comparação com as configurações anteriores, exibindo
menos suavização. À medida que aumentamos o valor de D0, permitimos que
detalhes mais finos da imagem se tornem visíveis, pois mais frequências mais altas
são retidas.

À medida que D0 aumenta, a diferença entre os três tipos de filtros (ideal,
Butterworth e Gaussiano) se torna aparente na forma como eles atenuam as
frequências próximas ao limite de corte:

O filtro ideal possui uma transição abrupta, cortando as frequências além do limite
de corte de forma abrupta e sem suavização.

O filtro Butterworth possui uma transição suave, controlada pela ordem do filtro, o
que afeta a nitidez da transição.
38

O filtro Gaussiano possui uma transição que segue uma distribuição gaussiana,
tornando a atenuação das frequências próximas ao limite de corte mais gradual e
suave.


5.     Efetue o mesmo que se pede no item 4, mas use o filtro passa-alta em vez do filtro passa-baixa.
import cv2
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './img/sinc_original.png'
sinc_original_menor_path = './img/sinc_original_menor.tif'
sinc_rot_path = './img/sinc_rot.png'
sinc_rot2_path = './img/sinc_rot2.png'
sinc_trans_path = './img/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path,
cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)
images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)',
'Transladada']

def fourier_spectrum(image):
    # Computa a transformada de Fourier 2D
    f = np.fft.fft2(image)

    # Centraliza as frequências baixas
    fshift = np.fft.fftshift(f)
    
    # Calcula a magnitude e aplica o logaritmo para melhor visualização
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

    def ideal_highpass_filter(image, cutoff):
        rows, cols = image.shape
        center_x, center_y = rows // 2, cols // 2
        filter = np.ones((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= cutoff:
                    filter[i, j] = 0
        return filter
    
def butterworth_highpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) **2)
            filter[i, j] = 1 / (1 + (cutoff / distance) ** (2 * order))
    return filter

def gaussian_highpass_filter(image, cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) **2)
            filter[i, j] -= np.exp(-(distance ** 2) / (2 * (cutoff **2)))
    return filter

def apply_filter(image, filter):
    # Aqui assumo que você está usando a Transformada de Fourier para aplicar o filtro.
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalizando a imagem resultante para o intervalo [0, 255]
    img_normalized = np.divide(img_back - np.min(img_back),
    np.max(img_back) - np.min(img_back)) * 255

    return img_normalized

cutoffs = [0.01, 0.05, 0.5]

for cutoff in cutoffs:
    for idx, image in enumerate(images):
        # Fourier
        spectrum = fourier_spectrum(image)

        # Criação dos filtros
        ideal_hp = ideal_highpass_filter(image, cutoff)
        butter_hp = butterworth_highpass_filter(image, cutoff)
        gaussian_hp = gaussian_highpass_filter(image, cutoff)

        # Aplicação dos filtros
        result_ideal = apply_filter(image, ideal_hp)
        result_butter = apply_filter(image, butter_hp)
        result_gaussian = apply_filter(image, gaussian_hp)
        
        # Exibição
        plt.figure(figsize=(20, 10))

        # Imagem original
        plt.subplot(4, 4, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'a) Imagem {titles[idx]}')
        plt.axis('off')

        # Espectro de Fourier
        plt.subplot(4, 4, 2)
        plt.imshow(spectrum, cmap='gray')
        plt.title('b) Espectro de Fourier')
        plt.axis('off')

        # Filtro passa-alta Ideal
        plt.subplot(4, 4, 3)
        plt.imshow(ideal_hp, cmap='gray')
        plt.title(f'c) Filtro Ideal Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Ideal
        plt.subplot(4, 4, 4)
        plt.imshow(result_ideal, cmap='gray')
        plt.title('d) Após Filtro Ideal')
        plt.axis('off')

        # Filtro passa-alta Butterworth
        plt.subplot(4, 4, 7)
        plt.imshow(butter_hp, cmap='gray')
        plt.title(f'c) Filtro Butterworth Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Butterworth
        plt.subplot(4, 4, 8)
        plt.imshow(result_butter, cmap='gray')
        plt.title('d) Após Filtro Butterworth')
        plt.axis('off')

        # Filtro passa-alta Gaussiano
        plt.subplot(4, 4, 11)
        plt.imshow(gaussian_hp, cmap='gray')
        plt.title(f'c) Filtro Gaussiano Passa-Alta (D0={cutoff})')
        plt.axis('off')

        # Resultado filtro Gaussiano
        plt.subplot(4, 4, 12)
        plt.imshow(result_gaussian, cmap='gray')
        plt.title('d) Após Filtro Gaussiano')
        plt.axis('off')
        
        # Ajusta o layout e mostra a figura
        plt.tight_layout()
        plt.show()
6.      Além dos filtros passa-baixa e passa-alta também existe o filtro passa-banda? Explique seu funcionamento e aplique um filtro passa-banda na imagem.

O filtro passa-banda permite a passagem de frequências dentro de uma faixa específica, rejeitando todas as outras frequências acima e abaixo dessa faixa. Isso é útil para realçar ou isolar determinadas características ou componentes de frequência em uma imagem ou sinal.

O funcionamento de um filtro passa-banda pode ser explicado da seguinte forma:
Especificações da Frequência de Corte: Em um filtro passa-banda, você precisa especificar duas frequências de corte: uma frequência de corte inferior (frequência de corte baixa, geralmente denotada como f1) e uma frequência de corte superior (frequência de corte alta, geralmente denotada como f2). Essas frequências definem a faixa de frequência que será permitida pelo filtro.

Transferência de Frequência: O filtro passa-banda funciona passando todas as frequências dentro da faixa especificada (f1 a f2) e atenuando (rejeitando) todas as frequências fora dessa faixa. A extensão e a forma da faixa de frequência dependem da configuração do filtro.
import numpy as np
import matplotlib.pyplot as plt

sinc_original_path = './img/sinc_original.png'
sinc_original_menor_path = './imagem/sinc_original_menor.png'
sinc_rot_path = './img/sinc_rot.png'
sinc_rot2_path = './img/sinc_rot2.png'
sinc_trans_path = './img/sinc_trans.png'

# Ler as imagens
sinc_original = cv2.imread(sinc_original_path, cv2.IMREAD_GRAYSCALE)
sinc_original_menor = cv2.imread(sinc_original_menor_path,
cv2.IMREAD_GRAYSCALE)
sinc_rot = cv2.imread(sinc_rot_path, cv2.IMREAD_GRAYSCALE)
sinc_rot2 = cv2.imread(sinc_rot2_path, cv2.IMREAD_GRAYSCALE)
sinc_trans = cv2.imread(sinc_trans_path, cv2.IMREAD_GRAYSCALE)

images = [sinc_original, sinc_rot, sinc_rot2, sinc_trans]
titles = ['Original', 'Rotacionada (40º)', 'Rotacionada (20º)',
'Transladada']

def fourier_spectrum(image):
    # Computa a transformada de Fourier 2D
    f = np.fft.fft2(image)
    # Centraliza as frequências baixas
    fshift = np.fft.fftshift(f)
    # Calcula a magnitude e aplica o logaritmo para melhor visualização
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def ideal_highpass_filter(image, cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= cutoff:
                filter[i, j] = 0
    return filter

def butterworth_highpass_filter(image, cutoff, order=2):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols): 
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) **2)
            filter[i, j] = 1 / (1 + (cutoff / distance) ** (2 * order))
    return filter

def gaussian_highpass_filter(image, cutoff):
    rows, cols = image.shape
    49
    center_x, center_y = rows // 2, cols // 2
    filter = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) **2)
            filter[i, j] -= np.exp(-(distance ** 2) / (2 * (cutoff **2)))
    return filter

def apply_filter(image, filter):
    # Aqui assumo que você está usando a Transformada de Fourier paraaplicar o filtro.
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift = fshift * filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalizando a imagem resultante para o intervalo [0, 255]
    img_normalized = np.divide(img_back - np.min(img_back),
    np.max(img_back) - np.min(img_back)) * 255
    return img_normalized

cutoffs = [0.01, 0.05, 0.5]

def ideal_bandpass_filter(image, Dl, Dh):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.uint8)
    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if Dl <= distance <= Dh:
                filter[x, y] = 1
    return filter

def apply_bandpass_filter(image, Dl, Dh):
    bandpass_filter = ideal_bandpass_filter(image, Dl, Dh)
    filtered_image = apply_filter(image, bandpass_filter)
    return filtered_image

# Aplicando o filtro
Dl = 10
Dh = 50
filtered_image = apply_bandpass_filter(sinc_original, Dl, Dh)

# Exibindo a imagem resultante
plt.figure(figsize=(10, 5))
plt.imshow(filtered_image, cmap='gray')
plt.title("Filtro passa-banda")
plt.axis('off')
plt.show()

## Aula 10 - 02/10 : Morfologia Matemática

Morfologia
 
- Erosão
- Dilatação
- Abertura
- Fechamento
Exercícios:

1. Implemente a erosão/dilatação utilizando os seguintes elementos estruturantes e utilize todas as imagens:

2. Implemente as operações de abertura e fechamento utilizando apenas o primeiro elemento estruturante do exercício acima. Considerando as imagens de b) a e) quais imagens seria mais interessante utilizar a abertura e quais o fechamento para remover os ruídos?

3. Qual sequência de operações poderia ser realizadas para que a imagem f) ficasse apenas com um retângulo branco ao centro? Implemente essas operações.

4. Qual(is) operações seriam necessárias para melhorar a imagem g)? Implemente essa(s) operação(ões).

5. Quais operações seriam necessárias para extrair apenas a borda da imagem h)? Implemente essas operações.

1. Implemente a erosão/dilatação utilizando os seguintes elementos estruturantes e utilize todas as imagens:

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definindo os paths das imagens
paths = [
    './img/{}.tif'.format(i)
    for i in ['fingerprint', 'Imagem1', 'Imagem2', 'morfologia1', 'morfologia2', 'noise_rectangle', 'rosto_perfil', 'text_gaps']
]

# Ler as imagens em escala de cinza
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]

# Definindo os elementos estruturantes
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
se_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Linha horizontal

# Iterando sobre cada imagem
for idx, img in enumerate(images):
    plt.figure(figsize=(15, 8))

    # Erosão
    img_eroded_disk = cv2.erode(img, se_disk)
    img_eroded_cross = cv2.erode(img, se_cross)
    img_eroded_line = cv2.erode(img, se_line)

    # Dilatação
    img_dilated_disk = cv2.dilate(img, se_disk)
    img_dilated_cross = cv2.dilate(img, se_cross)
    img_dilated_line = cv2.dilate(img, se_line)

    # Subplot 1: Imagem original
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Subplots 2-4: Imagens erodidas
    for i, eroded in enumerate([img_eroded_disk, img_eroded_cross, img_eroded_line]):
        plt.subplot(3, 4, i + 2)
        plt.imshow(eroded, cmap='gray')
        plt.title(f'Eroded ({["disk", "cross", "line"][i]})')
        plt.axis('off')

    # Subplots 5-7: Imagens dilatadas
    for i, dilated in enumerate([img_dilated_disk, img_dilated_cross, img_dilated_line]):
        plt.subplot(3, 4, i + 5)
        plt.imshow(dilated, cmap='gray')
        plt.title(f'Dilated ({["disk", "cross", "line"][i]})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

2. Implemente as operações de abertura e fechamento utilizando apenas o primeiro elemento estruturante do exercício acima. Considerando as imagens de b) a e) quais imagens seria mais interessante utilizar a abertura e quais o fechamento para remover os ruídos?

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definindo os paths das imagens
paths = [
    './img/{}.tif'.format(i)
    for i in ['fingerprint', 'Imagem1', 'Imagem2', 'morfologia1', 'morfologia2', 'noise_rectangle', 'rosto_perfil', 'text_gaps']
]

# Ler as imagens em escala de cinza
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]

# Elemento estruturante
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Iterando sobre cada imagem
for idx, img in enumerate(images):
    plt.figure(figsize=(15, 8))

    # Abertura
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se_disk)

    # Fechamento
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se_disk)

    # Subplot 1: Imagem original
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Subplot 2: Imagem após abertura
    plt.subplot(3, 3, 2)
    plt.imshow(img_opening, cmap='gray')
    plt.title('Opening')
    plt.axis('off')

    # Subplot 3: Imagem após fechamento
    plt.subplot(3, 3, 3)
    plt.imshow(img_closing, cmap='gray')
    plt.title('Closing')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

3. Qual sequência de operações poderia ser realizadas para que a imagem f) ficasse apenas com um retângulo branco ao centro? Implemente essas operações.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando a imagem f)
img_f = cv2.imread('./img/noise_rectangle.tif', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante (se_disk)
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Erosão para remover o ruído externo
img_eroded = cv2.erode(img_f, se_disk)

# Dilatação para restaurar o tamanho do retângulo ao centro
img_processed = cv2.dilate(img_eroded, se_disk)

# Exibindo a imagem original e a imagem processada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_f, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_processed, cmap='gray')
plt.title('Imagem Processada')
plt.axis('off')

plt.tight_layout()
plt.show()

4. Qual(is) operações seriam necessárias para melhorar a imagem g)? Implemente essa(s) operação(ões).

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando a imagem g)
img_g = cv2.imread('./img/rosto_perfil.tif', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante (se_disk)
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Dilatação para melhorar o texto
img_processed = cv2.dilate(img_g, se_disk)

# Exibindo a imagem original e a imagem processada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_g, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_processed, cmap='gray')
plt.title('Imagem Processada')
plt.axis('off')

plt.tight_layout()
plt.show()

5. Quais operações seriam necessárias para extrair apenas a borda da imagem h)? Implemente essas operações.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando a imagem h)
img_h = cv2.imread('./img/text_gaps.tif', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante (se_disk)
se_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Erosão para extrair a borda
img_eroded = cv2.erode(img_h, se_disk)

# Subtraindo a imagem erodida da imagem original para obter a borda
img_border = cv2.subtract(img_h, img_eroded)

# Exibindo a imagem original e a borda extraída
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_h, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_border, cmap='gray')
plt.title('Borda Extraída')
plt.axis('off')

plt.tight_layout()
plt.show()

## Aula 11 - 16/10: Segmentação Parte 1

1. Implementar limiarização, definir 

2. Implementar detector de bordas Canny.

    2.1. Aplicar o filtro de borramento (gaussiano) e verificar se o borramento melhora a detecção de bordas.

    2.2. Mudar os parametros T1 e T2 e avaliar a qualidade das bordas detectadas.


1. Implementar limiarização, definir 
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem
image = cv2.imread('./img/fingerprint.tif', 0)  # Certifique-se de substituir 'sua_imagem.jpg' pelo caminho da sua imagem

# Aplicar a limiarização
threshold_value = 128  # Você pode ajustar esse valor de acordo com suas necessidades
_, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# Exibir a imagem original e a imagem limiarizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Imagem Limiarizada')

plt.show()

2. Implementar detector de bordas Canny.

    2.1. Aplicar o filtro de borramento (gaussiano) e verificar se o borramento melhora a detecção de bordas.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('./img/fingerprint.tif', 0)  # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem

# 2.1. Aplicar filtro de borramento gaussiano
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # O segundo argumento é o tamanho do kernel gaussiano

# Aplicar o detector de bordas Canny na imagem original
canny_edges = cv2.Canny(image, 100, 200)  # Você pode ajustar os limiares conforme necessário

# Aplicar o detector de bordas Canny na imagem borradad
canny_edges_blurred = cv2.Canny(blurred_image, 100, 200)  # Novamente, ajuste os limiares conforme necessário

# Exibir as imagens
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')

plt.subplot(2, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Imagem Borradad (Gaussiano)')

plt.subplot(2, 2, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Detecção de Bordas (Canny) na Imagem Original')

plt.subplot(2, 2, 4)
plt.imshow(canny_edges_blurred, cmap='gray')
plt.title('Detecção de Bordas (Canny) na Imagem Borradad')

plt.show()

2.2. Mudar os parametros T1 e T2 e avaliar a qualidade das bordas detectadas.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('./img/fingerprint.tif', 0)  # Substitua 'sua_imagem.jpg' pelo caminho da sua imagem

# 2.1. Aplicar filtro de borramento gaussiano
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # O segundo argumento é o tamanho do kernel gaussiano

# Definir diferentes valores de T1 e T2
T1_values = [50, 100, 150]
T2_values = [100, 150, 200]

plt.figure(figsize=(12, 12))
plt.suptitle('Efeito dos parâmetros T1 e T2 na Detecção de Bordas', fontsize=16)

for i, (T1, T2) in enumerate(zip(T1_values, T2_values)):
    # Aplicar o detector de bordas Canny com os valores de T1 e T2
    canny_edges = cv2.Canny(image, T1, T2)

    plt.subplot(3, 3, i + 1)
    plt.imshow(canny_edges, cmap='gray')
    plt.title(f'T1={T1}, T2={T2}')

    plt.subplot(3, 3, i + 4)
    canny_edges_blurred = cv2.Canny(blurred_image, T1, T2)
    plt.imshow(canny_edges_blurred, cmap='gray')
    plt.title(f'T1={T1}, T2={T2} (Borradad)')

plt.show()
