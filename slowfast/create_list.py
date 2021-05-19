import os

f = open("test.txt","w")

target_dir = "/data/Peter/Data/Kinetics600_rescaled/"

classes = os.listdir(target_dir)
item_per_class = []
for counter, one_class in enumerate(sorted(classes)):
    counter_per_class = 0
    class_path = os.path.join(target_dir, one_class)
    image_paths = os.listdir(class_path)
    for image_path in sorted(image_paths):
        f.write(os.path.join(class_path, image_path))
        f.write(" ")
        f.write(str(counter))
        f.write("\n")
        counter_per_class+=1
    item_per_class.append(counter_per_class)

f.close()