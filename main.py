import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc

import cv2

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    x3 = (x2+1)
    for x in range(int(x1), int(x3)):
        coord = (int(y), int(x)) if is_steep else (int(x), int(y))
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points



"""co trzecia poczawszy od 0 potem 1 potem 2 pozycji"""
def sumowanie_pixel_proste(listoflist,img,n_odbiornikoww):
    n_odbiornikoww
    suma = 0
    tab = []
    tab2 = []

    for lines_nadajnik in listoflist:
        for line in lines_nadajnik:
            for point in line:
                x,y = point
                RGB = img.getpixel((x,y))
                suma = suma+RGB
            tab.append((suma / len(line))/255)
            suma = 0
        tab2.append(tab)
        tab = []
    return tab2

img = Image.open('aaa.png').convert('L')

start = 0

alfa = 2*np.pi/360
beta = np.pi
n_odbiornikow = 89
height, width = img.size
x0 = width/2
y0 = height/2
r = x0-10

fig = plt.figure()
fig.set_size_inches(16, 9, forward=True)

ax1 = fig.add_subplot(1, 3, 1)

ax1.imshow(img,cmap=plt.cm.gray)

n_nadajnikow = 0
tab_list = []
list_of_nadajnik = []


while (start < 2*np.pi - (alfa / 2)):
    """nadajnik"""
    x1 = x0 + r* np.cos(start)
    y1 = y0 + r* np.sin(start)


    c1 = plt.Circle((x1,y1), 5, color=(1, 0, 0))
    fig.add_subplot(131).add_artist(c1)
    """odbiornik"""

    for i in range(0,n_odbiornikow):
        x2 =  (x0 + r * np.cos((np.pi - (beta/2)) + start +((beta/(n_odbiornikow-1))*i)))
        y2 =  (y0 + r * np.sin((np.pi - (beta/2)) + start +((beta/(n_odbiornikow-1))*i)))
        c2 = plt.Circle((x2, y2), 5, color=(0, 1, 0))
        fig.add_subplot(131).add_artist(c2)
        line = get_line((x1,y1),(x2,y2))
        tab_list.append(line)
    start+=alfa
    n_nadajnikow = n_nadajnikow + 1
    list_of_nadajnik.append(tab_list)
    tab_list = []

print(len(list_of_nadajnik))

tab_pixel = []
tab_pixel = sumowanie_pixel_proste(list_of_nadajnik,img,n_odbiornikow)

ax1 = fig.add_subplot(1, 3, 2)

ax1.imshow(tab_pixel, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, aspect='auto')


ax1=fig.add_subplot(1,3,3)















mat = []
for i in range(width):
    row = []
    for j in range(height):
        row.append(0)
    mat.append(row)

i = 0
j = 0
wartosci_nadajnika = tab_pixel[i]
print(i)
for lines_nadajnik in list_of_nadajnik:
    for line in lines_nadajnik:
        for point in line:
            x,y = point
            mat[y][x] = mat[y][x] + wartosci_nadajnika[j]
        j = j + 1
    j = 0
    i = i + 1
    if (i == n_nadajnikow):
        break
    wartosci_nadajnika = tab_pixel[i]


max = 0
for i in range(width):
    for j in range(height):
        if mat[i][j] > max:
            max = mat[i][j]

for i in range(width):
    for j in range(height):
        mat[i][j] = mat[i][j] / max


ax1.imshow(mat, cmap=plt.get_cmap('gray'), vmax=1,vmin=0)

plt.show()

