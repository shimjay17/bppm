#train test 코드

from tqdm import tqdm
import torch
from utils import save_checkpoint, best_results, custom_CEL_min, custom_CEL_max
from torch.utils.tensorboard.writer import SummaryWriter
from torch.autograd import Variable

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, model_name, phases = ["train", "val"], classlabels = None):

    # assert "train" in phase
    
    x = 2

    main_results = []
    unit_results = []

    writer = SummaryWriter()

    for epoch in range(num_epochs):

        main_results.append(dict())
        unit_results.append(dict())

        main_preds = []
        unit_preds = []

        print(f"Epoch [{epoch+1}/{num_epochs}]")

        for phase in phases:

            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            running_main_loss = 0.0
            running_main_corrects = 0
            main_total = 0

            running_unit_loss = 0.0
            running_unit_corrects = 0
            unit_total = 0

            main_running_val_loss = 0.0
            main_running_val_corrects = 0
            main_running5_val_corrects = 0
            main_running10_val_corrects = 0
            main_total_val = 0

            main_train_loss = 0
            main_train_acc = 0
            main_val_loss = 0
            main_val_acc = 0
            main_val5_acc = 0
            main_val10_acc = 0


            unit_running_val_loss = 0.0
            unit_running_val_corrects = 0
            unit_running5_val_corrects = 0
            unit_running10_val_corrects = 0
            unit_total_val = 0

            unit_train_loss = 0
            unit_train_acc = 0
            unit_val_loss = 0
            unit_val_acc = 0
            unit_val5_acc = 0
            unit_val10_acc = 0

            combined_train_loss = 0
        
            # main_class_correct = torch.zeros(class_num()).cuda()
            # main_class_num = torch.zeros(class_num()).cuda()
            # main_class_acc = torch.zeros(class_num()).cuda()

            # unit_class_correct = torch.zeros(unit_combno_return()).cuda()
            # unit_class_num = torch.zeros(unit_combno_return()).cuda()
            # unit_class_acc = torch.zeros(unit_combno_return()).cuda()

            main_results[-1][phase] = dict(loss=[], acc=[])
            unit_results[-1][phase] = dict(loss=[], acc=[])
            
            if classlabels is not None:
                for label in classlabels:
                    main_results[-1][phase][label] = []

            if phase == 'train':
                pbar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), desc="Training", leave=False)
            if phase == 'val':
                pbar = tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader), desc="Validation", leave=False)

            for iteration, batch in pbar:

                inputs, labels_main, labels_unit, gt_name, camid = batch
                
                if phase == 'train':
                    inputs = inputs.to(device)

                    labels_main = labels_main.to(device)
                    labels_unit = labels_unit.to(device)

                    main_outputs, unit_outputs = model(inputs)
                    # main_outputs = model(inputs)

                else:
                    with torch.no_grad():
                        inputs = inputs.to(device)
                        labels_main = labels_main.to(device)
                        labels_unit = labels_unit.to(device)
                        main_outputs, unit_outputs = model(inputs)

                if x == 1: 
                    loss_main = custom_CEL_min(main_outputs, labels_main)
                    loss_unit = custom_CEL_min(unit_outputs, labels_unit)
                elif x == 2:
                    loss_main = custom_CEL_max(main_outputs, labels_main)
                    loss_unit = custom_CEL_max(unit_outputs, labels_unit)
                elif x == 3:
                    loss_main = criterion(main_outputs, labels_main)
                    loss_unit = criterion(unit_outputs, labels_unit)

                loss = loss_main+loss_unit
                loss = Variable(loss, requires_grad=True)
                # loss = loss_unit

                _, main_preds = torch.max(main_outputs, 1)
                _, unit_preds = torch.max(unit_outputs, 1)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase =='train':
                    running_main_loss += loss_main.item() * inputs.size(0)
                    running_main_corrects += torch.sum(main_preds == labels_main.data)
                    main_total += labels_main.size(0)

                    main_train_loss = running_main_loss / main_total
                    main_train_acc = running_main_corrects.double() / main_total

                    running_unit_loss += loss_unit.item() * inputs.size(0)
                    running_unit_corrects += torch.sum(unit_preds == labels_unit.data)
                    unit_total += labels_unit.size(0)

                    unit_train_loss = running_unit_loss / unit_total
                    unit_train_acc = running_unit_corrects.double() / unit_total

                    combined_train_loss = main_train_loss + unit_train_loss
                                        
                elif phase == 'val':
                    
                    tmp,main_top_idx = torch.sort(main_outputs.detach(),dim=-1,descending=True)
                    Labels_main = labels_main.unsqueeze(-1)
                    main_top_5 = main_top_idx[:,:5]
                    main_top_5_acc = main_top_5 == Labels_main
                    main_top_5_crt = main_top_5_acc.sum()
                    main_running5_val_corrects += main_top_5_crt

                    main_top_10 = main_top_idx[:,:10]
                    main_top_10_acc = main_top_10 == Labels_main
                    main_top_10_crt = main_top_10_acc.sum()
                    main_running10_val_corrects += main_top_10_crt

                    main_running_val_loss += loss_main.item() * inputs.size(0)
                    main_running_val_corrects += torch.sum(main_preds == labels_main.data)
                    main_total_val += labels_main.size(0)

                    main_val_loss = main_running_val_loss / float(main_total_val)
                    main_val_acc = main_running_val_corrects.double() / main_total_val
                    main_val5_acc = main_running5_val_corrects.double() / main_total_val
                    main_val10_acc = main_running10_val_corrects.double() / main_total_val

                    tmp, unit_top_idx = torch.sort(unit_outputs.detach(), dim=-1, descending=True)
                    Labels_unit = labels_unit.unsqueeze(-1)
                    unit_top_5 = unit_top_idx[:, :5]
                    unit_top_5_acc = unit_top_5 == Labels_unit
                    unit_top_5_crt = unit_top_5_acc.sum()
                    unit_running5_val_corrects += unit_top_5_crt

                    unit_top_10 = unit_top_idx[:, :10]
                    unit_top_10_acc = unit_top_10 == Labels_unit
                    unit_top_10_crt = unit_top_10_acc.sum()
                    unit_running10_val_corrects += unit_top_10_crt

                    unit_running_val_loss += loss_unit.item() * inputs.size(0)
                    unit_running_val_corrects += torch.sum(unit_preds == labels_unit.data)
                    unit_total_val += labels_unit.size(0)

                    unit_val_loss = unit_running_val_loss / float(unit_total_val)
                    unit_val_acc = unit_running_val_corrects.double() / unit_total_val
                    unit_val5_acc = unit_running5_val_corrects.double() / unit_total_val
                    unit_val10_acc = unit_running10_val_corrects.double() / unit_total_val

                    combined_val_loss = main_val_loss + unit_val_loss

            if phase == 'train':
                writer.add_scalar('Main/Train/Loss', main_train_loss, epoch)
                writer.add_scalar('Main/Train/Acc', main_train_acc, epoch)
                writer.add_scalar('Unit/Train/Loss', unit_train_loss, epoch)
                writer.add_scalar('Unit/Train/Acc', unit_train_acc, epoch)
                writer.add_scalar('Combined_Loss/Train', combined_train_loss, epoch)
                
            elif phase == 'val':
                writer.add_scalar('Main/Val/Loss', main_val_loss, epoch)
                writer.add_scalar('Main/Val/Acc', main_val_acc, epoch)
                writer.add_scalar('Main/Val/Acc5', main_val5_acc, epoch)
                writer.add_scalar('Main/Val/Acc10', main_val10_acc, epoch)
                writer.add_scalar('Unit/Val/Loss', unit_val_loss, epoch)
                writer.add_scalar('Unit/Val/Acc', unit_val_acc, epoch)
                writer.add_scalar('Unit/Val/Acc5', unit_val5_acc, epoch)
                writer.add_scalar('Unit/Val/Acc10', unit_val10_acc, epoch)
                writer.add_scalar('Combined_Loss/Val', combined_val_loss, epoch)

            writer.flush()

            if phase == 'val':
                main_results[-1][phase]["loss"].append(main_val_loss)
                main_results[-1][phase]["acc"].append(main_val_acc)

                unit_results[-1][phase]["loss"].append(unit_val_loss)
                unit_results[-1][phase]["acc"].append(unit_val_acc)

                # results[-1][phase]["acc5"].append(val5_acc)
                # results[-1][phase]["acc10"].append(val10_acc)

            if phase == 'train':
                print(f"Main Block Train Loss: {main_train_loss:.4f}, Main Block Train Acc: {main_train_acc:.4f}")
                print(f"Unit Block Train Loss: {unit_train_loss:.4f}, Unit Block Train Acc: {unit_train_acc:.4f}")

            elif phase == 'val':
                print(f"Main Block Val Loss: {main_val_loss:.4f}, Main Block Val Acc: {main_val_acc:.4f}, Main Block Val5 Acc: {main_val5_acc:.4f}, Main Block Val10 Acc: {main_val10_acc:.4f}")
                print(f"Unit Block Val Loss: {unit_val_loss:.4f}, Unit Block Val Acc: {unit_val_acc:.4f}, Unit Block Val5 Acc: {unit_val5_acc:.4f}, Unit Block Val10 Acc: {unit_val10_acc:.4f}")

        best_val_acc = best_results(unit_results)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, best_val_acc, model_name)
        
    writer.close()