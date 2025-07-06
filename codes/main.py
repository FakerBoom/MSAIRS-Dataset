from dataload import *
from model import *
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

eopchs =  CFG['epochs']+1

def get_data(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    sticker_id = list(set([i['sticker'] for i in data]))
    random.shuffle(sticker_id) # 随机打乱
    train,valid = [],[]

    
    for item in data:
        if item['sticker'] in sticker_id[:int(len(sticker_id)*0.9)]:
            train.append(item)
        else:
            valid.append(item)
        
    return train,valid

# 判读是否存在train.json和valid.json，如果不存在则生成
if not os.path.exists('/home/ycshi/sticker-intent1/data/train.json') or not os.path.exists('/home/ycshi/sticker-intent1/data/valid.json'):
    train,valid = get_data('/home/ycshi/sticker-intent1/data/rewrite.json')

    # 将train和valid的数据存储为json文件
    with open('/home/ycshi/sticker-intent1/data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False)
    with open('/home/ycshi/sticker-intent1/data/valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid, f, ensure_ascii=False)

else:
    # 如果已经存在了，直接从json文件中读取
    with open('/home/ycshi/sticker-intent1/data/train.json', encoding='utf-8') as f:
        train = json.load(f)
    with open('/home/ycshi/sticker-intent1/data/valid.json', encoding='utf-8') as f:
        valid = json.load(f)

def train_model(model, train_loader,valid_loader, optimizer):

    max_emo = [0,0,0,0,0, 0, 0]
    max_intent = [0,0,0,0,0]
    maxadd = [0,0,0,0,0]
    for epoch in range(1,eopchs):
        model.train()
        train_loss = 0
        emo_acc = 0
        emo_acc1 = 0
        intent_acc = 0
        
        # 实时更新进度条和loss
        with tqdm(total=len(train_loader), desc=f'Train Epoch {epoch}/{eopchs}', unit='batch', ascii=True) as pbar:
            for batch in train_loader:
                context_ids, context_masks, token_type_ids,sticker_txt_ids, stickertxt_masks,stickertxt_type_ids , sticker_features, emo_lable , intent_lable, emo_t, emo_s = batch
                context_ids = context_ids.to(device)
                context_masks = context_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                sticker_txt_ids = sticker_txt_ids.to(device)
                stickertxt_masks = stickertxt_masks.to(device)
                stickertxt_type_ids = stickertxt_type_ids.to(device)
                sticker_features = sticker_features.to(device)
                emo_lable = emo_lable.to(device)
                intent_lable = intent_lable.to(device)
                emo_s = emo_s.to(device)
                emo_t = emo_t.to(device)

                loss, emo_classifier_output, emo_classifier_outputs, intent_classifier_output = model(context_ids, context_masks, token_type_ids,sticker_txt_ids ,stickertxt_masks,stickertxt_type_ids ,sticker_features, emo_lable , intent_lable, emo_t, emo_s)
                emo_predict = torch.argmax(emo_classifier_output, dim=1)
                emo_predicts = torch.argmax(emo_classifier_outputs, dim=1)
                intent_predict = torch.argmax(intent_classifier_output, dim=1)
                #emo_acc += torch.sum(emo_predict == emo_lable).item()
                emo_acc += torch.sum(emo_predict == emo_t).item()
                emo_acc1 += torch.sum(emo_predicts == emo_s).item()
                intent_acc += torch.sum(intent_predict == intent_lable).item()
                
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': '{:.3f}'.format(loss.item())})
                pbar.update(1)
            #print(f'Epoch {epoch}/{eopchs} train_loss: {train_loss/len(train_loader):.3f} emo_acc: {emo_acc/len(train_loader.dataset):.3f} intent_acc: {intent_acc/len(train_loader.dataset):.3f}')
            print(f'Epoch {epoch}/{eopchs} train_loss: {train_loss/len(train_loader):.3f} emo_acc: {emo_acc/len(train_loader.dataset):.3f} emo_acc1: {emo_acc1/len(train_loader.dataset):.3f} intent_acc: {intent_acc/len(train_loader.dataset):.3f}')
        # 每个epoch结束后，计算在验证集上的loss和acc
        model.eval()
        valid_loss = 0
        val_emo_acc = 0
        val_emo_acc1 = 0
        val_intent_acc = 0
        val_emo_f1 = 0
        val_emo_f11 = 0
        val_intent_f1 = 0
        # 实时更新进度条和loss
        
        with tqdm(total=len(valid_loader), desc=f'valid Epoch {epoch}/{eopchs}', unit='batch', ascii=True) as pbar:               
            with torch.no_grad():
                for batch in valid_loader:
                    context_ids, context_masks, token_type_ids,sticker_txt_ids,stickertext_masks,stickertext_type_ids , sticker_features, emo_lable , intent_lable, emo_t, emo_s = batch
                    context_ids = context_ids.to(device)
                    context_masks = context_masks.to(device)
                    token_type_ids = token_type_ids.to(device)
                    sticker_txt_ids = sticker_txt_ids.to(device)
                    stickertext_masks = stickertext_masks.to(device)
                    stickertext_type_ids = stickertext_type_ids.to(device)
                    sticker_features = sticker_features.to(device)
                    emo_lable = emo_lable.to(device)
                    intent_lable = intent_lable.to(device)
                    emo_s = emo_s.to(device)
                    emo_t = emo_t.to(device)

                    loss, emo_classifier_output, emo_classifier_outputs,intent_classifier_output = model(context_ids, context_masks, token_type_ids,sticker_txt_ids ,stickertext_masks,stickertext_type_ids ,sticker_features, emo_lable , intent_lable, emo_t, emo_s)
                    emo_predict = torch.argmax(emo_classifier_output, dim=1)
                    emo_predicts = torch.argmax(emo_classifier_outputs, dim=1)
                    intent_predict = torch.argmax(intent_classifier_output, dim=1)
                    #val_emo_acc += torch.sum(emo_predict == emo_lable).item()
                    val_emo_acc += torch.sum(emo_predict == emo_t).item()
                    val_emo_acc1 += torch.sum(emo_predicts == emo_s).item()
                    val_intent_acc += torch.sum(intent_predict == intent_lable).item()
                    #val_emo_f1 += f1_score(emo_lable.cpu(), emo_predict.cpu(), average='weighted')
                    val_emo_f1 += f1_score(emo_t.cpu(), emo_predict.cpu(), average='weighted')
                    val_emo_f11 += f1_score(emo_s.cpu(), emo_predicts.cpu(), average='weighted')
                    val_intent_f1 += f1_score(intent_lable.cpu(), intent_predict.cpu(), average='weighted')
                    valid_loss += loss.item()
                #print(f'Epoch {epoch}/{eopchs} valid_loss: {valid_loss/len(valid_loader):.3f} emo_acc: {val_emo_acc/len(valid_loader.dataset):.3f} intent_acc: {val_intent_acc/len(valid_loader.dataset):.3f}')
                print(f'Epoch {epoch}/{eopchs} valid_loss: {valid_loss/len(valid_loader):.3f} emo_acc: {val_emo_acc/len(valid_loader.dataset):.3f} emo_acc1: {val_emo_acc1/len(valid_loader.dataset):.3f} intent_acc: {val_intent_acc/len(valid_loader.dataset):.3f}')
                #print(f'Epoch {epoch}/{eopchs} valid_loss: {valid_loss/len(valid_loader):.3f} emo_f1: {val_emo_f1/len(valid_loader):.3f} intent_f1: {val_intent_f1/len(valid_loader):.3f}')
                print(f'Epoch {epoch}/{eopchs} valid_loss: {valid_loss/len(valid_loader):.3f} emo_f1: {val_emo_f1/len(valid_loader):.3f} emo_f11: {val_emo_f11/len(valid_loader):.3f} intent_f1: {val_intent_f1/len(valid_loader):.3f}')
                if (val_emo_acc/len(valid_loader.dataset)) > max_emo[1]:
                    max_emo[0] = epoch
                    max_emo[1] = val_emo_acc/len(valid_loader.dataset)
                    max_emo[2] = val_intent_acc/len(valid_loader.dataset)
                    max_emo[3] = val_emo_f1/len(valid_loader)
                    max_emo[4] = val_intent_f1/len(valid_loader)
                    max_emo[5] = val_emo_acc1/len(valid_loader.dataset)
                    max_emo[6] = val_emo_f11/len(valid_loader)
                    
                if (val_intent_acc/len(valid_loader.dataset)) > max_intent[2]:
                    max_intent[0] = epoch
                    max_intent[1] = val_emo_acc/len(valid_loader.dataset)
                    max_intent[2] = val_intent_acc/len(valid_loader.dataset)
                    max_intent[3] = val_emo_f1/len(valid_loader)
                    max_intent[4] = val_intent_f1/len(valid_loader)
                    
                if ((val_emo_acc/len(valid_loader.dataset))+(val_intent_acc/len(valid_loader.dataset))) > (maxadd[1]+maxadd[2]):
                    maxadd[0] = epoch
                    maxadd[1] = val_emo_acc/len(valid_loader.dataset)
                    maxadd[2] = val_intent_acc/len(valid_loader.dataset)
                    maxadd[3] = val_emo_f1/len(valid_loader)
                    maxadd[4] = val_intent_f1/len(valid_loader)

    print(f'max_emo: {max_emo}')
    print(f'max_intent: {max_intent}')
    print(f'maxadd: {maxadd}')
   
if __name__ == '__main__':
    import os

    train_dataset = MyDataset(train)
    train_loader = DataLoader(train_dataset, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True, num_workers=CFG['num_workers'])
    valid_dataset = MyDataset(valid)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=CFG['num_workers'])

    device = torch.device(CFG['device'])
    model = module().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    train_model(model, train_loader,valid_loader, optimizer)