from pytorch_lightning import Logger

class PrintLogger(Logger):
    def log_metrics(self, metrics, step):
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}", end=" ")
        print()