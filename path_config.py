import os

root_dir = os.path.dirname(__file__)
predictor_dir = os.path.join(root_dir, 'predictor')
img_dir = os.path.join(root_dir, 'img')

caffe_model = os.path.join(root_dir, 'MobileNetSSD/MobileNetSSD_deploy.caffemodel')
proto_text = os.path.join(root_dir, 'MobileNetSSD/MobileNetSSD_deploy.prototxt.txt')

test_img = os.path.join(img_dir, 'cars.jpg')

print(caffe_model)
