import os
path = '/Users/ruihangzhang/Desktop/test'
files = os.listdir(path)
id = 1

for file in files:
    name = f"pixiv{id}.png"
    id += 1 
    
    old = os.path.join(path, file)
    new = os.path.join(path, name)

    if os.path.isfile(old):
        os.rename(old, new)

