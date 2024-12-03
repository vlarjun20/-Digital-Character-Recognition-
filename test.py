from ultralytics import YOLO
model=YOLO("new yolo.pt")

model.predict(source="chumma.jpeg",show=True)
