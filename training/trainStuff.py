# Core libraries
import torch
import os

# PyTorch utilities
from torch.optim import AdamW

# Progress bar
from tqdm import tqdm





#--- training function ---

#note our load save pattern is: load on gpu and save on gpu

def trainer(model, dataloader, path, epochs=3, lr=2e-5, device="cuda"):


    #--- model set up ---
    if(os.path.exists(rf"{path}")):
        model.load_state_dict(torch.load(rf"{path}", weights_only=True))
        print("loaded in model!")

    model.to(device) #gpu!

    optimizer = AdamW(model.parameters(), lr=lr)


    #--- training looops ---
    for epoch in range(epochs):

        # sets model to training mode
        model.train()
        
        # good visual for the terminal to see progress
        loop = tqdm(dataloader, leave=True)
        
        
        total_loss = 0

        #--- loop through batches ---
        for batch in loop:

            #--- reset gradient ---
            optimizer.zero_grad()

            #--- move input tensors to cpu or gpu ---
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["labels"].to(device)


            #--- run the model ---
            logits, loss = model(input_ids, attention_mask, extra_features, labels)

            #--- loss + optimize the model ---
    
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            
            #--- update progress bar ---
            loop.set_description(f"Epoch {epoch+1}")
            
            #show current batch loss
            loop.set_postfix(loss=loss.item())


        #--- Save trained model ---

        torch.save(model.state_dict(), path)
        
        # Print the average loss for the entire epoch
        print(f"Epoch {epoch+1} finished. Average loss: {total_loss/len(dataloader):.4f}")





#note our load save pattern is: load on gpu and save on gpu
def evaluater(model, dataloader, path, device="cuda"):

    #--- model set up ---

    if(os.path.exists(rf"{path}")):
        model.load_state_dict(torch.load(rf"{path}", weights_only=True))
        print("loaded in model!")

    model.to(device) #gpu!

    #evaluation mode
    model.eval() 


    total_loss = 0
    all_preds = []
    all_labels = []


    with torch.no_grad():  # no gradients needed

        #--- progress bar ---
        loop = tqdm(dataloader, leave=True)


        for batch in loop:

            #--- move input tensors to cpu or gpu ---
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["labels"].to(device)


            #--- run the model ---
            logits, loss = model(input_ids, attention_mask, extra_features, labels)

            #--- loss ---
            total_loss += loss.item()

            #--- predictiction ---
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # collect predictions and labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            #--- update progress bar ---

            loop.set_postfix(loss=loss.item())


    #--- get all predictionas and labels ---
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    #--- total accuracy and average loss ---
    accuracy = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

    return all_preds, all_labels