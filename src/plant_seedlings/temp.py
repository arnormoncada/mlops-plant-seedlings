from data import plant_seed_dataloader
from model import MyAwesomeModel

train_dataloader = plant_seed_dataloader(data_path="data/raw", batch_size=32)
model = MyAwesomeModel()

for img, target in train_dataloader:
    print(img.shape, target.shape)
    
    model(img)
    break