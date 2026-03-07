from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from dataset import Data
from trainStuff import trainer, evaluater
import os
import torch
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RobertaLmodels import RobertaClassifierWithExtra

             
def train_evaluate_Model():
    
    #--- Debug ---
    print(torch.version.cuda)       
    print(torch.cuda.is_available())


    #--- train/evaluation loop ---
    load_dotenv()
    path = os.getenv("model")
    print(path)

    model =  RobertaClassifierWithExtra()
    train_loader, val_loader = Data(16)

    trainer(model, train_loader, path)

    all_preds, all_labels = evaluater(model, val_loader, path, device="cuda")

    #--- Move tensors to CPU and convert to numpy ---
    y_pred = all_preds.numpy()
    y_true = all_labels.numpy()


    #--- Print Metrics ---

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)



if __name__ == "__main__":
    train_evaluate_Model()
