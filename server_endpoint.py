from training.dataset import search
from RobertaLmodels import RobertaClassifierWithExtra
from transformers import RobertaTokenizer
import torch
import os



class InputText(BaseModel):
    text: str



@app.post("/predict")
def predict(data: InputText):
    
    #--- model ---
    os.load_dotenv()
    path = os.getenv("model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "roberta-base"
    model =  RobertaClassifierWithExtra()
    
    if(os.path.exists(rf"{path}")):
        model.load_state_dict(torch.load(rf"{path}", weights_only=True))
        print("loaded in model!")

    model.to(device) #gpu!


    #--- Collect input ----

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True, max_length=384)

    query = data.text[:500]
    bm25_max, bm25_avg, bm25_variance, links = search(query)
    

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    bm25 = torch.tensor([bm25_max, bm25_avg, bm25_variance], dtype=torch.float32).unsqueeze(0).to(device)

    #--- forward pass ---
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, bm25)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()


    return {"prediction": pred, "confidence": confidence, "links": links}


