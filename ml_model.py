import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils import logger


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AstronomyAnomalyDetector:
    def __init__(self, model_dir="models", contamination=0.05):
        # Yetki hatasını (PermissionError) önlemek için güvenli yol belirleme
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_dir = os.path.join(base_dir, model_dir)

        try:
            os.makedirs(candidate_dir, exist_ok=True)
            self.model_dir = candidate_dir
        except Exception:
            # Yazma izni yoksa kullanıcı ana dizinine güvenli klasör aç
            self.model_dir = os.path.join(os.path.expanduser("~"), ".planetx_models")
            os.makedirs(self.model_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae_path = os.path.join(self.model_dir, "ae_weights.pth")

        # PyTorch Autoencoder
        self.ae_model = ConvAutoencoder().to(self.device)
        self.is_trained = False

        if os.path.exists(self.ae_path):
            try:
                self.ae_model.load_state_dict(torch.load(self.ae_path, map_location=self.device))
                self.ae_model.eval()
                self.is_trained = True
                logger.info(f"Autoencoder ağırlıkları yüklendi: {self.ae_path}")
            except Exception as e:
                logger.error(f"Model yükleme hatası: {e}")

        # Isolation Forest
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(contamination=contamination, random_state=42)

    def fit_predict(self, features):
        if len(features) == 0:
            return np.array([]), np.array([])
        scaled = self.scaler.fit_transform(features)
        preds = self.iforest.fit_predict(scaled)
        scores = self.iforest.decision_function(scaled)
        return preds, scores

    def extract_patches(self, image, patch_size=32):
        h, w = image.shape
        patches = []
        coords = []

        img_min, img_max = np.nanmin(image), np.nanmax(image)
        norm_img = (image - img_min) / (img_max - img_min + 1e-8)
        norm_img = np.nan_to_num(norm_img, nan=0.0)

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = norm_img[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
                coords.append((x, y))

        if len(patches) == 0:
            return np.array([]), []

        patches_tensor = np.array(patches, dtype=np.float32)[:, np.newaxis, :, :]
        return patches_tensor, coords

    def train_autoencoder(self, patches_tensor, epochs=3, batch_size=32, lr=0.001):
        dataset = TensorDataset(torch.tensor(patches_tensor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.ae_model.parameters(), lr=lr)

        self.ae_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = self.ae_model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        torch.save(self.ae_model.state_dict(), self.ae_path)
        self.is_trained = True

    def detect_anomalies_ae(self, patches_tensor, coords, threshold_percentile=95):
        if not self.is_trained or len(patches_tensor) == 0:
            return [], []

        self.ae_model.eval()
        criterion = nn.MSELoss(reduction='none')

        dataset = TensorDataset(torch.tensor(patches_tensor))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        mse_errors = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.ae_model(inputs)
                loss = criterion(outputs, inputs).mean(dim=[1, 2, 3])
                mse_errors.extend(loss.cpu().numpy())

        mse_errors = np.array(mse_errors)
        mean_err = np.mean(mse_errors)
        std_err = np.std(mse_errors)

        threshold = mean_err + 1.2 * std_err if std_err > 1e-6 else np.percentile(mse_errors, threshold_percentile)

        anomalies = []
        for i, err in enumerate(mse_errors):
            if err >= threshold:
                x, y = coords[i]
                anomalies.append({
                    "x": x + 16,
                    "y": y + 16,
                    "score": float(err)
                })

        return anomalies, mse_errors