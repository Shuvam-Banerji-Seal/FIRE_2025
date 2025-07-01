import torch 
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
from huggingface_hub import login
login("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")

quant_config=BitsAndBytesConfig(load_in_8bit=True)
use_quant_config=False
device='cuda' if torch.cuda.is_available() else "cpu"
# model_id="meta-llama/Llama-3.2-3B-Instruct"
# model_id='meta-llama/Llama-3.2-1B'
model_id='google/gemma-3-1b-it'
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id,padding_side='right')
llm_model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                               torch_dtype=torch.bfloat16,
                                               attn_implementation='eager',
                                               quantization_config=quant_config if use_quant_config else None).to(device)

def prompt_generator(data:dict):
    text=data['Text']
    comment=data['comment']
    datatype=data['relevance']

    prompt=f''' 
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an Ai classification system that Read and understand the given Text and it's comment in input then analyze that wether the given comment is relevant or not in 
context of given Text . if given comment is relevant in context of Text give output as 1 is it irrelevant than 0
NO EXPLANATIONS is required . YOU must choose from One of following Classes:
Labelled class: O
or Labelled class: 1
Ensure Strictly that output is either 0 or 1 based on relevancy of given comment on given Text.


### Input:
Text-:{text}

Comment-:{comment}

### Response:
Labelled class: {datatype}{tokenizer.eos_token}'''
    return prompt


tokenizer.pad_token = tokenizer.eos_token
class Instruction_datast(Dataset):
    def __init__(self,csv_file,tokenizer,Max_length):
        
        self.file=pd.read_csv(csv_file)
        self.data= self.file[['Text','comment','relevance']].to_dict(orient='records')

        all_response=self._full_response(data=self.data)
        self.encoded=[tokenizer.encode(response) for response in all_response]
 
        if Max_length == None:
            self.max_length=self._longest_length()
        else:
            self.max_length=Max_length
        

        encoded_text=tokenizer(all_response,return_tensors='pt',padding='max_length',truncation=True,max_length=self.max_length)
        encoded_id=encoded_text['input_ids']
        encoded_mask=encoded_text['attention_mask']
             
        self.input_ids=encoded_id[:,:-1]
        self.input_mask=encoded_mask[:, :-1]
        self.target_ids=encoded_id[: ,1:]
        self.target_mask=encoded_mask[:, 1:]


        self.target_compare = encoded_id[:, 1:].clone()
        answer_texts = [" "+str(item["relevance"]) + tokenizer.eos_token for item in self.data]
        self.answer_token_ids =[tokenizer.encode(ans, add_special_tokens=False) for ans in answer_texts]

        
        for i in range(self.target_compare.shape[0]):
            full = self.target_compare[i]  # shape: [seq_len
            response=torch.tensor(self.encoded[i])
            answer_ids = torch.tensor(self.answer_token_ids[i])

            position=response.shape[0]-answer_ids.shape[0]-1
            is_match = torch.equal(full[position:position + answer_ids.shape[0]],answer_ids)

            if is_match:
                mask = torch.full_like(full, -100)
                mask[position:position + answer_ids.shape[0]] = full[position:position + answer_ids.shape[0]]
                self.target_compare[i] = mask 
            else:
                print('not found')
        


    def _longest_length(self):
        return max((len(encoding) for encoding in self.encoded),default=0)

    def _full_response(self,data:list[dict]):
        full_response=[]
        for dict in data:
            prompt=prompt_generator(data=dict)
            full_response.append(prompt)
        return full_response
    
    def __len__(self):
        return len(self.file) 

    def __getitem__(self,index):
        return self.input_ids[index],self.target_compare[index]
    

dataset_test=Instruction_datast(csv_file='qna_test.csv',tokenizer=tokenizer,Max_length=None)
dataset_train=Instruction_datast(csv_file='qna_train.csv',tokenizer=tokenizer,Max_length=None)
dataset_val=Instruction_datast(csv_file='qna_validation.csv',tokenizer=tokenizer,Max_length=None)

num_workers = 0
batch_size = 4

torch.manual_seed(123)


train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_loader = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(input_batch)
            logits = output.logits 
            
            preds = logits.argmax(dim=-1) 
            mask = (target_batch != -100)  

            correct = (preds == target_batch) & mask
            total_correct += correct.sum().item()
            total_count += mask.sum().item()

    return total_correct / total_count if total_count > 0 else float('nan')




def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    output = model(input_batch)  
    logits=output.logits
    if torch.isnan(logits).any():
        print("⚠️ logits contain NaN!")
    if torch.isinf(logits).any():
        print("⚠️ logits contain Inf!")
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  
        target_batch.view(-1),             
        ignore_index=-100                  
    )
    return loss

def calc_loss_accuracy(dataloader,device,model,num_batches=None):
    total_loss=0
    if num_batches == None:
        num_batches=len(dataloader)
    else:
        num_batches= min(num_batches,len(dataloader))
    for i,(input_batch,target_batch) in enumerate(dataloader):
        if i<num_batches:
            # print("target min:", target_batch.min().item(), "max:", target_batch.max().item())
            loss= calc_loss_batch(input_batch=input_batch,target_batch=target_batch,model=model,device=device)
            total_loss+= loss.item()
        else:
            break
    return total_loss/num_batches

def evaluate_model(model, train_loader, vali_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_accuracy(train_loader, device, model, eval_iter)
        vali_loss = calc_loss_accuracy(vali_loader, device, model, eval_iter)
        train_acc = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_acc = calc_accuracy_loader(vali_loader, model, device, eval_iter)
    model.train()
    return train_loss, vali_loss, train_acc, val_acc



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter,scheduler,best_val_accuracy=0.75):
   
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

   
    for epoch in range(num_epochs):
        model.train()  
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() 
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() 
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1


            if global_step % eval_freq == 0:
                train_loss, val_loss, train_acc, val_acc = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} | "
                      f"Train acc {train_acc*100:.2f}%, Val acc {val_acc*100:.2f}%")
                if val_acc > best_val_accuracy:
                    if (train_acc - val_acc) <0.05:
                        best_val_accuracy = val_acc 
                        torch.save({
                            'model_state_dict': llm_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, f'qna_model{int(train_acc*100)}_{int(val_acc*100)}_.pt')
                        print(f"Model saved with improved accuracy: {val_acc:.4f}")

    return train_losses, val_losses, track_tokens_seen

import time

start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(llm_model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 3
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.2 * total_steps)  
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

train_losses, val_losses, tokens_seen = train_model_simple(
    llm_model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=40, eval_iter=25,scheduler=scheduler
)
  
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")