import os
from PIL import Image
import matplotlib.pyplot as plt

train_folder = 'train2017'
val_folder = 'val2017'

def count_images_in_folder(folder_path):

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg'))]
    return len(image_files)

num_train_images = count_images_in_folder(train_folder)
num_val_images = count_images_in_folder(val_folder)

print(f"Количество изображений в train2017: {num_train_images}")
print(f"Количество изображений в val2017: {num_val_images}")

def get_image_sizes(folder_path):
    sizes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                sizes.append((width, height))
    return sizes

train_sizes = get_image_sizes(train_folder)
val_sizes = get_image_sizes(val_folder)

train_widths, train_heights = zip(*train_sizes)
val_widths, val_heights = zip(*val_sizes)

plt.figure(figsize=(10, 6))

plt.scatter(train_widths, train_heights, color='blue', alpha=0.5, label='Train')

plt.scatter(val_widths, val_heights, color='red', alpha=0.5, label='Val')

plt.title("Размеры изображений в датасетах Train и Val")
plt.xlabel("Ширина (pixels)")
plt.ylabel("Высота (pixels)")
plt.legend()

plt.show()