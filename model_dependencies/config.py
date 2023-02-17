import os

cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = '/home/dezmon/Documents/Object2/train'
VALID_DATASET_PATH = '/home/dezmon/Documents/Object2/vaild'
TEST_DATASET_PATH = '/home/dezmon/Documents/Object2/test'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'hardwarev3.tflite'
CLASSES = ['Duck', 'Red Chips', 'Green Chips','Green Pedestal','Green Pedestal_Tilt','Red Pedestal','Red Pedestal_Tilt','Duck_Left','Duck_Right','Duck_Front','Duck_Back']
EPOCHS = 20
BATCH_SIZE = 1