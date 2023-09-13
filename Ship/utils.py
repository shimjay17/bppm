#checkpoint 세이브 코드
#checkpoint 로드 코드
#best result 갱신시 true 리턴 하는 코드

import os
import glob
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
from matplotlib.colors import Normalize
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image, ImageDraw, ImageFont
import re
import collections
from configloader import configloader

(learning_rate,
    num_epochs,
    batch_size,
    weight_decay,
    class_weights,
    train_set_path,
    test_set_path,
    block_dict_path,
    unit_dict_path,
    relationship_dict_path,
    mothership_dict_path,
    results_dict_path,
    checkpoint_path,
    snapshot_directory,
    json_directory,
    jpg_directory,
    results_directory) = configloader()
    

def save_checkpoint(state, is_best_val, model_name):

    torch.save(state, os.path.join('/home/admin/workspace/bppm/Ship/param', model_name + ".pt"))

    if is_best_val:
        print('best val saved!')
        shutil.copyfile(os.path.join('/home/admin/workspace/bppm/Ship/param', model_name + ".pt"), os.path.join('/home/admin/workspace/bppm/Ship/param', model_name + "_bval.pt"))

def load_checkpoint(net, optimizer, filename, is_cuda=True, remove_module=False, add_module=False):

    if os.path.isfile(filename):
        checkpoint = torch.load(filename) if is_cuda else torch.load(filename, map_location=lambda storage, loc: storage)
        model_state = net.state_dict()

        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if remove_module:
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        if add_module:
            state_dict = {'module.' + k: v for k, v in state_dict.items() }

        for k, v in state_dict.items():
            if k in model_state and v.size() == model_state[k].size():
                # print("[INFO] Loading param %s with size %s into model."%(k, ','.join(map(str, model_state[k].size()))))
                pass
            else:
                # print("Size in model is ", v.size(), filename)
                print("[WARNING] Could not load params %s in model." % k)

        pretrained_state = {k: v for k, v in state_dict.items() if
                            k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        net.load_state_dict(model_state)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("[WARNING] Could not find params file %s." % filename)


def best_results(results):

    has_val = 'val' in results[-1] and len(results[-1]['val']) > 0

    val_loss_results = [results[i]["val"]["loss"][-1] for i in range(len(results) - 1)] if has_val else []
    best_val_loss = has_val and (results[-1]["val"]["loss"][-1] <= (np.min(val_loss_results) if len(val_loss_results) > 0 else np.inf))
    val_acc_results = [results[i]["val"]["acc"][-1].cpu() for i in range(len(results))] if has_val else[]
    best_val_acc = has_val and (results[-1]["val"]["acc"][-1] >= np.max(val_acc_results))

    return best_val_acc


def save_matrix(matrix_main, matrix_unit, matrix_comb, filter):

    if filter == 0:
        mode = '(unfiltered)'
    elif filter == 1:
        mode = '(filtered)'

    # Create a list of matrices and their corresponding names
    matrices = [(matrix_main, 'main'), (matrix_unit, 'unit'), (matrix_comb, 'comb')]

    # Loop over the matrices and save the plots
    for matrix, name in matrices:
        # Create a confusion matrix display
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

        # Plot the confusion matrix
        disp.plot(include_values=True,
                  cmap='viridis', 
                  xticks_rotation='horizontal',
                  values_format='.2f')

        plt.title(f'Confusion Matrix {name.capitalize()} {mode}')
        plt.savefig(f'matrix_{name}_{mode}.png')
        plt.clf()


def confusion_matrix(l, o):
    matrix = np.zeros((5,5))
    L = len(l)
    for i in range(L):
        label = int(l[i])
        pred = int(o[i])
        matrix[label][pred] = matrix[label][pred] + 1
    return matrix


def custom_CEL_min(logits, labels):
    B = logits.shape[0]-1
    loss_matrix = torch.zeros(logits.shape[0], logits.shape[0])
    for i in range(logits.shape[0]):
        for j in range(logits.shape[0]):
            loss_matrix[i, j] = nn.functional.cross_entropy(logits[j].unsqueeze(0), labels[i].unsqueeze(0))
    
    row_sum = torch.sum(loss_matrix, dim=1)

    n_loss = row_sum - torch.diagonal(loss_matrix)

    p_loss = torch.diagonal(loss_matrix)

    row_loss = -n_loss/B + p_loss

    losses = torch.clamp(row_loss, min=0, max=None)

    losses_u = torch.mean(losses)

    # print(losses_u)
    
    return losses_u

def custom_CEL_max(logits, labels):
    factor = 0.2

    batch_size = logits.shape[0]
    loss_matrix = torch.zeros(batch_size, batch_size).to(logits.device)

    for i in range(batch_size):
        loss_matrix[i] = nn.functional.cross_entropy(logits, labels[i].unsqueeze(0).expand(batch_size), reduction='none')

    # Create a mask that excludes the diagonal
    mask = torch.ones_like(loss_matrix) - torch.eye(batch_size, device=logits.device)

    # Apply the mask to the loss_matrix
    masked_losses = loss_matrix * mask

    # Compute the maximum loss of each row excluding diagonal elements
    n_loss, _ = torch.max(masked_losses, dim=1)

    p_loss = torch.diagonal(loss_matrix)

    row_loss = p_loss - n_loss - factor

    losses = torch.clamp(row_loss, min=0, max=None)

    losses_u = torch.mean(losses)

    return losses_u


def metric_calc(labels_main, main_top_idx, main_preds, labels_unit, unit_top_idx, unit_preds, comb_top_idx, comb_preds, main_loss, unit_loss, inputs):
    

    main_running_test_loss = 0.0
    main_running_test_corrects = 0
    main_running5_test_corrects = 0
    main_running10_test_corrects = 0
    main_running100_test_corrects = 0
    main_running_test_loss_avg = 0.0
    main_running_test_corrects_avg = 0
    main_running5_test_corrects_avg = 0
    main_running10_test_corrects_avg = 0
    main_running100_test_corrects_avg = 0
    main_total_test = 0

    unit_running_test_loss = 0.0
    unit_running_test_corrects = 0
    unit_running5_test_corrects = 0
    unit_running10_test_corrects = 0
    unit_running100_test_corrects = 0
    unit_total_test = 0
    
    comb_running_test_corrects = 0
    comb_running5_test_corrects = 0
    comb_running10_test_corrects = 0
    comb_running100_test_corrects = 0
    comb_total_test = 0

    main_results = []
    unit_results = []
    comb_results = []

    main_results.append(dict())
    unit_results.append(dict())
    comb_results.append(dict())
            
    main_results[-1]["test"] = dict(loss=[], acc=[])
    unit_results[-1]["test"] = dict(loss=[], acc=[])
    comb_results[-1]["test"] = dict(loss=[], acc=[])

    for i in range(inputs.size(0)):
        Labels_main = labels_main.unsqueeze(-1)
        main_top_5 = main_top_idx[:,:5]
        main_top_5_acc = main_top_5 == Labels_main
        main_top_5_crt = main_top_5_acc.sum()
        main_running5_test_corrects += main_top_5_crt

        main_top_10 = main_top_idx[:,:10]
        main_top_10_acc = main_top_10 == Labels_main
        main_top_10_crt = main_top_10_acc.sum()
        main_running10_test_corrects += main_top_10_crt

        main_top_100 = main_top_idx[:,:100]
        main_top_100_acc = main_top_100 == Labels_main
        main_top_100_crt = main_top_100_acc.sum()
        main_running100_test_corrects += main_top_100_crt

        main_running_test_loss += main_loss.item() * inputs.size(0)
        main_running_test_corrects += torch.sum(main_preds == labels_main.data)
        main_total_test += labels_main.size(0)

        Labels_unit = labels_unit.unsqueeze(-1)
        unit_top_5 = unit_top_idx[:,:5]
        unit_top_5_acc = unit_top_5 == Labels_unit
        unit_top_5_crt = unit_top_5_acc.sum()
        unit_running5_test_corrects += unit_top_5_crt

        unit_top_10 = unit_top_idx[:,:10]
        unit_top_10_acc = unit_top_10 == Labels_unit
        unit_top_10_crt = unit_top_10_acc.sum()
        unit_running10_test_corrects += unit_top_10_crt

        unit_top_100 = unit_top_idx[:,:100]
        unit_top_100_acc = unit_top_100 == Labels_unit
        unit_top_100_crt = unit_top_100_acc.sum()
        unit_running100_test_corrects += unit_top_100_crt

        unit_running_test_loss += unit_loss.item() * inputs.size(0)
        unit_running_test_corrects += torch.sum(unit_preds == labels_unit.data)
        unit_total_test += labels_unit.size(0)         

        Labels_comb = labels_unit.unsqueeze(-1)
        comb_top_5 = comb_top_idx[:,:5]
        comb_top_5_acc = comb_top_5 == Labels_comb
        comb_top_5_crt = comb_top_5_acc.sum()
        comb_running5_test_corrects += comb_top_5_crt

        comb_top_10 = comb_top_idx[:,:10]
        comb_top_10_acc = comb_top_10 == Labels_comb
        comb_top_10_crt = comb_top_10_acc.sum()
        comb_running10_test_corrects += comb_top_10_crt

        comb_top_100 = comb_top_idx[:,:100]
        comb_top_100_acc = comb_top_100 == Labels_comb
        comb_top_100_crt = comb_top_100_acc.sum()
        comb_running100_test_corrects += comb_top_100_crt

        comb_running_test_corrects += torch.sum(comb_preds == labels_unit.data)
        comb_total_test += labels_unit.size(0)

    
    main_test_loss = main_running_test_loss / main_total_test
    main_test_acc = main_running_test_corrects.double() / main_total_test
    main_test5_acc = main_running5_test_corrects.double() / main_total_test
    main_test10_acc = main_running10_test_corrects.double() / main_total_test
    main_test100_acc = main_running100_test_corrects.double() / main_total_test

    unit_test_loss = unit_running_test_loss / unit_total_test
    unit_test_acc = unit_running_test_corrects.double() / unit_total_test
    unit_test5_acc = unit_running5_test_corrects.double() / unit_total_test
    unit_test10_acc = unit_running10_test_corrects.double() / unit_total_test
    unit_test100_acc = unit_running100_test_corrects.double() / unit_total_test

    comb_test_loss = unit_running_test_loss / comb_total_test
    comb_test_acc = comb_running_test_corrects.double() / comb_total_test
    comb_test5_acc = comb_running5_test_corrects.double() / comb_total_test
    comb_test10_acc = comb_running10_test_corrects.double() / comb_total_test
    comb_test100_acc = comb_running100_test_corrects.double() / comb_total_test

    # low_class_values, low_class_postitions = torch.topk(class_acc, 5, largest=False)


    main_results[-1]["test"]["loss"].append(main_test_loss)
    main_results[-1]["test"]["acc"].append(main_test_acc)
    unit_results[-1]["test"]["loss"].append(unit_test_loss)
    unit_results[-1]["test"]["acc"].append(unit_test_acc)
    comb_results[-1]["test"]["loss"].append(comb_test_loss)
    comb_results[-1]["test"]["acc"].append(comb_test_acc)
    # results[-1]["test"]["acc5"].append(test5_acc)
    # results[-1]["test"]["acc10"].append(test10_acc)

    print(f"Main Test Loss: {main_test_loss:.4f}, Main Test Acc: {main_test_acc:.4f}, Main Test5 Acc: {main_test5_acc:.4f}, Main Test10 Acc: {main_test10_acc:.4f}, Main Test100 Acc: {main_test100_acc:.4f}")
    print(f"Unit Test Loss: {unit_test_loss:.4f}, Unit Test Acc: {unit_test_acc:.4f}, Unit Test5 Acc: {unit_test5_acc:.4f}, Unit Test10 Acc: {unit_test10_acc:.4f}, Unit Test100 Acc: {unit_test100_acc:.4f}")
    print(f"Comb Test Loss: {comb_test_loss:.4f}, Comb Test Acc: {comb_test_acc:.4f}, Comb Test5 Acc: {comb_test5_acc:.4f}, Comb Test10 Acc: {comb_test10_acc:.4f}, Comb Test100 Acc: {comb_test100_acc:.4f}")

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ensemble(result_path):
    all_predictions = {i: "None" for i in range(1, 13)}
    img_name = {}
    step = {i: "None" for i in range(1, 13)}


    folder_path = results_dict_path
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            match = re.search(r'CAM=(0[1-9]|1[0-2])', file_name) 
            if match:
                cam_id = int(match.group(1))  # Extract camera id from filename
                file_path = os.path.join(folder_path, file_name)

                with open(file_path, 'r') as file:
                    pred_dict = json.load(file)
                    #get first 10 values of pred_dict
                    all_predictions[cam_id] = list(pred_dict.items())[:-2]
                    step[cam_id] = list(pred_dict.items())[-2][1].items()
                    img_name[cam_id] = list(pred_dict.values())[-1]

    ensemble_groups = {
        "1": [1, 2, 7, 8],
        "2": [2, 3, 8, 9],
        "3": [3, 4, 9, 10],
        "4": [4, 5, 10, 11],
        "5": [5, 6, 11, 12]
    }

    ensemble_results = {}
    step_ensemble_results = {}

    for group_name, camera_ids in ensemble_groups.items():
        combined_predictions = []
        step_predictions = []
        used = []
        not_used = []
        img_load = []

        for cam_id in camera_ids:
            #ignore if cam_id is not in all_predictions

            if all_predictions[cam_id] == 'None':
                not_used.append(cam_id)
                continue
            else: 
                used.append(cam_id)
                combined_predictions.extend(all_predictions[cam_id])
                step_predictions.extend(step[cam_id])


        if np.size(combined_predictions) == 0:
            print("No results to print")
        else:
            # if label is equal, add probability
            combined_predictions = sorted(combined_predictions, key=lambda x: x[0])
            combined_predictions = [list(i) for i in combined_predictions]
            for i in range(len(combined_predictions)-1):
                if combined_predictions[i][0] == combined_predictions[i+1][0]:
                    combined_predictions[i][1] = combined_predictions[i][1] + combined_predictions[i+1][1]
                    combined_predictions[i+1][1] = 0
            combined_predictions = [i for i in combined_predictions if i[1] != 0]

            step_predictions = sorted(step_predictions, key=lambda x: x[0])
            step_predictions = [list(i) for i in step_predictions]
            for i in range(len(step_predictions)-1):
                if step_predictions[i][0] == step_predictions[i+1][0]:
                    step_predictions[i][1] = step_predictions[i][1] + step_predictions[i+1][1]
                    step_predictions[i+1][1] = 0
            step_predictions = [i for i in step_predictions if i[1] != 0]

            #sort by probability
            combined_predictions = sorted(combined_predictions, key=lambda x: x[1], reverse=True)
            step_predictions = sorted(step_predictions, key=lambda x: x[1], reverse=True)

            #get top 5
            top5 = combined_predictions[:5]

            #softmax combined_predictions

            comb_probabilities = [item[1] for item in top5]
            step_probabilities = [item[1] for item in step_predictions]
            

            # Apply softmax
            comb_softmax_probs = softmax(comb_probabilities)
            softmax_probs = softmax(step_probabilities)


            # Update the data with softmax probabilities
            for i, item in enumerate(top5):
                item[1] = comb_softmax_probs[i]

            for i, item in enumerate(step_predictions):
                item[1] = softmax_probs[i]

            ensemble_results[group_name] = top5
            step_ensemble_results[group_name] = step_predictions

            print(ensemble_results)
            # print_results(ensemble_results[group_name], group_name, camera_ids, img_name, used) #######
            print_results(ensemble_results[group_name], step_ensemble_results[group_name], group_name, camera_ids, img_name, used, result_path) #######

        
# def print_results(ensemble_results, group_name, camid, img_name, used):
def print_results(ensemble_results, step_ensemble_results, group_name, camid, img_name, used,result_path):
    # Load the mapping dictionary
    with open(unit_dict_path) as json_file:
        unit_block_dict = json.load(json_file)

    # Reverse the dictionary for easy access
    reversed_dict = {v: k for k, v in unit_block_dict.items()}

    label_height = 18  # Adjust based on your needs and the font size
    font = ImageFont.load_default()

    x_offset, y_offset = 20, 0
    canvas1 = Image.new('RGB', (160, label_height))
    for cam in camid:
        #if in used
        draw_pred = ImageDraw.Draw(canvas1)

        if cam in used:
            draw_pred.text((x_offset, y_offset), str(cam), font=font, fill="green")
        else:
            draw_pred.text((x_offset, y_offset), str(cam), font=font, fill="grey")
        x_offset += 80



    y_offset = 0

    canvas2 = Image.new('RGB', (160, label_height * len(ensemble_results)))

    for label, score in ensemble_results:
        label = int(label)
        
        original_label = reversed_dict[label]

        # Format the label with its probability rounded to two decimal places
        text = f"{original_label}: {score:.2f}"

        # Render the formatted text onto the image
        draw_pred = ImageDraw.Draw(canvas2)
        draw_pred.text((10, y_offset), text, font=font, fill="white")
        y_offset += label_height

        
    canvas3 = Image.new('RGB', (160, 64))

    # Offsets for pasting images onto the canvas
    x_offset, y_offset = 0, 0

    # Counter to track the number of images added to the canvas
    count = 0

    # Total images to be loaded
    total_images = len(used)

    for cam in used:
        img = img_name[cam]

        # Use glob to find the first matching file
        matching_files = glob.glob(os.path.join(result_path, img))
        if matching_files:
            image = Image.open(matching_files[0])
        else:
            print(f"No matching file found for image: {img}")
            continue

        image = image.convert('RGB')
        image = image.resize((80, 32))
        
        # Paste the image onto the canvas at the correct position
        canvas3.paste(image, (x_offset, y_offset))
        
        # Update offsets and counter
        count += 1
        x_offset += 80
        if count % 2 == 0:  # Move to the next row after every 2 images
            x_offset = 0
            y_offset += 32

        # Break if we've loaded four images
        if count > total_images:
            break

        canvas4 = Image.new('RGB', (160, label_height))
        for label, score in step_ensemble_results:

            # Format the label with its probability rounded to two decimal places
            step = '0'
            if label == '0':
                step = "Empty"
            elif label == '1':
                step = "Step"
            elif label == '2':
                step = "None"

            text = f"{step}: {score:.2f}"

            # Render the formatted text onto the image
            draw_pred = ImageDraw.Draw(canvas4)
            draw_pred.text((10, 0), text, font=font, fill="white")

        combined_image = Image.new('RGB', (160, 64 + canvas1.height + canvas2.height + canvas4.height))
        combined_image.paste(canvas3, (0, 0))
        combined_image.paste(canvas1, (0, 64))
        combined_image.paste(canvas2, (0, 64 + canvas1.height))
        combined_image.paste(canvas4,(0, 64 + canvas1.height + canvas2.height))

        sdirectory = os.path.dirname(result_path)
        save_directory = os.path.join(sdirectory,'prediction')
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        combined_image.save(os.path.join(save_directory , 'ensemble-' + group_name +'-results.png'))
        
    # return combined_image
