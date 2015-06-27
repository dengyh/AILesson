from PIL import Image
from random import randint
visit = []
img = Image.open('avatar_bin.jpg')
xlen, ylen = img.size
for i in xrange(xlen * ylen / 10):
    x = randint(0, xlen - 1)
    y = randint(0, ylen - 1)
    while (x, y) in visit:
        x = randint(0, xlen - 1)
        y = randint(0, ylen - 1)
    visit.append((x, y))
    img.putpixel((x, y), 255 - img.getpixel((x, y)))
img.save('avatar_noise.jpg')
img.show()