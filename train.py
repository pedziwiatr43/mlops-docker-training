import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import wandb

def run_training(config):
    # --- Initialisierung ---
    wandb.init(project=config.get("wandb_project", "docker-ml-demo"), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # --- Dummy-Daten erstellen ---
    X = torch.randn(500, 10)
    y = (X.sum(dim=1) > 0).float()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # --- Einfaches Modell ---
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # --- Training Loop ---
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # --- Modell speichern ---
    checkpoint_path = os.path.join(config["checkpoint_dir"], "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    wandb.finish()
