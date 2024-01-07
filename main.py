from ultralytics import YOLO
import torch

# Set to your desired GPU number
torch.cuda.set_device(0) 

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Train the model
    results = model.train(data='VOC.yaml', epochs=3, imgsz=640, device='gpu')

    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
