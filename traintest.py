import time
import torch 
import tqdm
from utils import plot_attention
from utils import transform_index_to_letter, calc_edit_distance
from utils import index2letter, letter2index, LETTER_LIST

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(device)
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Train')
    start_time = time.time()
    running_loss = 0
    running_purplex = 0.0
    mode = 'train'
    random_rate = max(1 - 0.1*(epoch//5),0.5)

    
    # 0) Iterate through your data loader
    for i, (x, x_len, y, y_len) in enumerate(train_loader):
        
        # 1) Send the inputs to the device
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # 2) Pass your inputs, and length of speech into the model.
        predictions, attentions = model(x, x_len, y, mode=mode, random_rate = random_rate)
        attentions = attentions.permute(1, 0, 2)
        
        # 3) Generate a mask based on target length. This is to mark padded elements
        # so that we can exclude them from computing loss.
        # Ensure that the mask is on the device and is the correct shape.

        y_len = y_len.clone().detach().to(device)
        max_len = torch.max(y_len)
        mask = (torch.arange(0, max_len).repeat(y_len.size(0), 1).to(device) < y_len.unsqueeze(1).expand(y_len.size(0), max_len)).int() # fill this out
        mask = mask.to(device)

        # 4) Make sure you have the correct shape of predictions when putting into criterion
        # loss = criterion(# fill this out)
        loss = criterion(predictions.view(-1, predictions.size(2)), y.view(-1))
        # Use the mask you defined above to compute the average loss
        # masked_loss = # fill this out
        masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)

        # 5) backprop

        masked_loss.backward()
        
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        # When computing Levenshtein distance, make sure you truncate prediction/target

        optimizer.step()

        current_loss = masked_loss.item()
        current_purplex = torch.exp(masked_loss).item()
        running_loss += current_loss
        running_purplex += current_purplex

        #########################################

        model_name = 'hw4p2_model4-4'
        
        # Optional: plot your attention for debugging
        # plot_attention(attentions)
        if i == 0:
            plot_attention(attentions[0,:x_len[0],:x_len[0]], 'plots/attention_{}_train_loadertest_epoch_{}_batch_{}'.format(model_name, epoch, i))

        batch_bar.set_postfix(
            epoch=epoch,
            loss='{:.04f}'.format(float(running_loss / (i + 1))),
            lr = '{:.04f}'.format(float(optimizer.param_groups[0]['lr']))
        )

        batch_bar.update()

    batch_bar.close()
    
    end_time = time.time()
    print("Finished Epoch: {}\ttrain loss: {:.4f}\ttrain perplex: {:.4f}\ttime: {:.4f}".format(epoch,\
          running_loss/len(train_loader), running_purplex/len(train_loader), end_time - start_time))

    # wandb.log({'epoch': epoch,
    #     'train_loss': float(running_loss / len(train_loader)),
    #     'lr': float(optimizer.param_groups[0]['lr'])})

    return running_loss/len(train_loader)

def eval(model, eval_loader, epoch):
    with torch.no_grad():
        model.eval()
        model.to(device)
        batch_bar = tqdm(total=len(eval_loader), dynamic_ncols=True, position=0, leave=False, desc='Eval')
        preds = []
        start_time = time.time()
        running_loss = 0
        running_purplex = 0.0
        running_distance = 0
        num_seq = 0
        mode = 'val'
        
        for i, (x, lx, y, ly) in enumerate(eval_loader):
            x, y = x.to(device), y.to(device)
            predictions, attentions = model(x, lx, y, mode=mode)
            attentions = attentions.permute(1, 0, 2)

            pred_text = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy())
            y_text = transform_index_to_letter(y.detach().cpu().numpy())

            running_distance += calc_edit_distance(pred_text, y_text) #, 1 if i==0 else 0) 
            num_seq += len(pred_text) # batch_size

            model_name = 'hw4p2_model4-4'
            if i == 0:
                plot_attention(attentions[0,:lx[0],:lx[0]], 'plots/attention_{}_eval_loadertest_epoch_{}_batch_{}'.format(model_name, epoch, i))

            batch_bar.set_postfix(
                epoch=epoch,
                loss='{:.04f}'.format(float(running_loss / (i + 1))),
                lr = '{:.04f}'.format(float(optimizer.param_groups[0]['lr']))
            )

            batch_bar.update()

        batch_bar.close()

        end_time = time.time()

        wandb.log({'epoch': epoch,
            'edit_dist': running_distance/num_seq})

        print("Finished Epoch: {}\tedit distance: {:.4f}\ttime: {:.4f}".format(epoch, running_distance/num_seq, end_time - start_time))
        
    return running_distance/num_seq

#%%
def test(model, test_loader):
    model.eval()
    model.to(device)
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    preds = []
    start_time = time.time()
    mode = 'test'

    for i, (x, lx) in enumerate(test_loader):
        x = x.to(device)
        
        with torch.no_grad():
            predictions, _ = model(x, lx, mode=mode)

        pred_text = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy())

        preds.extend(pred_text)
        batch_bar.update()

    batch_bar.close()

    return preds
