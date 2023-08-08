import os
i = 1
for filename in os.listdir():
    if filename.endswith('.jpg'):
        new_name ='damaged_orange_'+str(i) + '.jpg'
        os.rename(filename, new_name)
        i += 1