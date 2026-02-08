import os
import random 
from pathlib import Path
from tabnanny import verbose

from sympy.printing.pretty.stringpict import line_width
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from ultralytics import YOLO

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"drone_dataset"
datasetPath = Path(DATA_PATH)
DATA_YAML = Path(DATA_PATH) / "data.yaml"
SAVED_WEIGHTS = datasetPath.parent / "runs" / "detect" / "runs" / "detect" / "drone" / "weights" / "best.pt"

if not datasetPath.exists():
    raise FileNotFoundError(f"Data path {DATA_PATH} does not exist.")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Data.yaml path {DATA_YAML} does not exist.")


class DroneDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        imgDir = self.root / split / "images"
        labelDir = self.root / split / "labels"
        if not imgDir.exists():
            imgDir = self.root / split 
            labelDir = self.root / split 
        for p in imgDir.glob("*.jpg"):
            labelPath = labelDir / (p.stem + ".txt")
            label = 1 if (labelPath.exists() and labelPath.read_text().strip()) else 0
            self.samples.append((str(p), label))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path)
        if img.mode in ("P", "PA"):
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
trainTF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
valTF = trainTF

trainDS = DroneDataset(datasetPath, split="train", transform=trainTF)
valDS = DroneDataset(datasetPath, split="valid", transform=valTF)
trainLoader = DataLoader(trainDS, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
valLoader = DataLoader(valDS, batch_size=16, shuffle=False, num_workers=0)

if __name__ == "__main__":
    print("This model is using: ", device)
    print(f"Data.yaml path: {DATA_YAML}")
    if SAVED_WEIGHTS.exists():
        print(f"using saved model: {SAVED_WEIGHTS}")
        model = YOLO(str(SAVED_WEIGHTS))
        bestWeights = SAVED_WEIGHTS
    else:
        print("No saved model found. Training.")
        model = YOLO("yolo26n.pt")
        results = model.train(
            data = str(DATA_YAML),
            epochs = 50,
            patience = 15,
            imgsz = 640,
            batch = 16,
            device = "cuda:0",
            project = str(datasetPath.parent / "runs" / "detect"),
            name = "drone",
            exist_ok = True,
            verbose = True,
        )

        bestWeights = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best weights saved to: {bestWeights}")
        
    validImgDir = datasetPath / "valid" / "images"
    if not validImgDir.exists():
        validImgDir = datasetPath / "valid"
    allValidImages = list(validImgDir.glob("*.jpg"))
    
    if not allValidImages:
        print("No validation images found")
    else:
        nShow = min(random.randint(4,9), len(allValidImages))
        selectedPaths = random.sample(allValidImages, nShow)
        bestModel = YOLO(str(bestWeights))
        predResults = bestModel.predict(
            source = [str(p) for p in selectedPaths],
            save=False,
            line_width=2,
        )
        for i, r in enumerate(predResults):
            plotImg = r.plot()
            cv2.imshow(f"Validation {i+1}/{nShow}", plotImg)
        print(f"Showing {nShow} random validation images")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #cap = cv2.VideoCapture("/dev/video10")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open cam")
    else:
        webcamModel = YOLO(str(bestWeights))
        print("Webcam drone detection. Press q to exit")
        cv2.namedWindow("Drone detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Drone detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        smoothed = None
        lastConf = 0.0
        alpha = 0.4
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = webcamModel.predict(frame, conf=0.2, verbose=False)
            det = None
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                # pick best: highest confidence
                idx = boxes.conf.argmax().item()
                xyxy = boxes.xyxy[idx].cpu().numpy()
                conf = float(boxes.conf[idx])
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                det = (cx, cy, w, h, conf)

            if det is not None:
                cx, cy, w, h, conf = det
                if smoothed is None:
                    smoothed = (cx, cy, w, h)
                else:
                    scx, scy, sw, sh = smoothed
                    smoothed = (
                        alpha * cx + (1 - alpha) * scx,
                        alpha * cy + (1 - alpha) * scy,
                        alpha * w + (1 - alpha) * sw,
                        alpha * h + (1 - alpha) * sh,
                    )
                last_conf = conf
            # else: keep previous smoothed (and last_conf) so path doesn't vanish

            if smoothed is not None:
                scx, scy, sw, sh = smoothed
                x1 = int(scx - sw / 2)
                y1 = int(scy - sh / 2)
                x2 = int(scx + sw / 2)
                y2 = int(scy + sh / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"drone {last_conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Drone detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()