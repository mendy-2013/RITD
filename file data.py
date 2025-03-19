import os
paths = r'/home/dell/DB-master/datasets/TD_TR/TR400/train_images'  # 储存图片的文件夹路径
f = open('images1.txt', 'w')
filenames = os.listdir(paths)  # 读取图片名称
for filename in filenames:
    out = filename
    f.write(out + '\n')
f.close()