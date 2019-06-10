from flask import request
from fastai import *
from fastai.vision import *
import sys

path = Path('/data/cloth_categories')

   
def predict(imagePath):

    classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']
    single_img_data = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(),size=150).normalize(imagenet_stats)
    learn = create_cnn(single_img_data, models.resnet34)
    learn.load('/content/Fine-Grained-Clothing-Classification/data/cloth_categories/models/stage-1_sz-150')
    _, _, losses = learn.predict(open_image(imagePath))
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    return predictions

@app.route('/retailGyan/api/v1.0/predict', methods=['POST'])
def predict_task():
    if not request.json or not 'imgPath' in request.json:
        abort(400)
    imgPath =  request.json['imgPath']
    preds = predict(imgPath)
    return preds, 201


