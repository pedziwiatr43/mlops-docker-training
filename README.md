# ML Training Pipeline mit Docker und Weights & Biases

Dieses Projekt demonstriert eine einfache Machine-Learning-Trainingspipeline mit
PyTorch, Docker und Weights & Biases (W&B).  
Ziel ist es zu zeigen, wie ein reproduzierbares ML-Training in einer vollständig
containerisierten Umgebung aufgebaut werden kann.

## Projektziele

- Aufbau einer ML-Trainingspipeline mit PyTorch  
- Nutzung von Docker für reproduzierbare Umgebungen  
- Logging und Monitoring mit Weights & Biases  
- Strukturierter, nachvollziehbarer Projektaufbau  

## Projektstruktur

```
.
├── main.py
├── train.py
├── config.yaml
├── requirements.txt
├── Dockerfile
└── models/
```

## Installation (lokal)

### Repository klonen
```
git clone <repo-url>
cd <projektname>
```

### Python-Abhängigkeiten installieren
```
pip install -r requirements.txt
```

### Training starten
```
python main.py --config config.yaml
```

## Training mit Docker

### Docker-Image bauen
```
docker build -t ml-training:latest .
```

### Container starten

Offline:
```
docker run --rm -e WANDB_MODE=offline ml-training:latest
```

Online:
```
docker run --rm -e WANDB_API_KEY=<dein_key> ml-training:latest
```

## Weights & Biases

Konfiguration über `config.yaml`:
```
wandb_project: "my-project"
```

## Ausgabe

- trainiertes Modell: `models/model.pt`  
- W&B Logs offline/online

## Reproduzierbarkeit

- Docker stellt gleiche Umgebung überall sicher  
- funktioniert auf Codespaces ohne Änderungen  

## Zweck der Abgabe

Studierendenprojekt zur Demonstration einer einfachen ML-Docker-Pipeline.

## Autor

Mateusz
