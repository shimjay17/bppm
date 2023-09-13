from tqdm import tqdm
import torch
import os
from countmachine import class_num, unit_combno_return
from utils import save_matrix, print_results, ensemble
import numpy as np
import random
import json
from torch.nn.functional import softmax
from step_detection.test import step_test


def test(model, dataloader, optimizer, criterion, device, relationship_dict_path, result_path, desc, num_epochs=1):

    with open(relationship_dict_path, 'r') as f:
        relationship = json.load(f)

    a = 2
    b = 1

    filter = 0

    num_iterations = 1

    for epoch in range(num_epochs):


        model.eval()

        pbar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=desc, leave=False)

        with torch.no_grad():
            for iteration, batch in pbar:
                
                inputs, gt_name, camid = batch
                
                inputs = inputs.to(device)
                
                optimizer.zero_grad()

                main_outputs, unit_outputs = model(inputs)

                main_tmp = torch.zeros_like(unit_outputs)

                for i in range(num_iterations):
                    if filter == 0:
                        _, main_preds = torch.max(main_outputs, 1)

                        for t in range(main_outputs.shape[0]):
                            for main_class, main_output in enumerate(main_outputs[t]):
                                unit_classes = relationship[str(main_class)]

                                for unit_class in unit_classes:
                                    main_tmp[t, unit_class] = main_output.item()

                        comb_outputs = b * unit_outputs + a * main_tmp

                    elif filter == 1:
                        main_selected_outputs = torch.full_like(main_outputs, -1000)
                        unit_selected_outputs = torch.full_like(unit_outputs, -1000)

                        if desc == "test2" or desc == "test3" or desc == "test":
                            main_selected_outputs[:, labels_main] = main_outputs[:, labels_main]
                            unit_selected_outputs[:, labels_unit] = unit_outputs[:, labels_unit]

                        else:
                            # Search Space 최적화 기능 시뮬래이션, 제대로 사용하려면 random indices를 변환한 스케줄 정보를 가지고 오게 바꿔야함
                            for t in range(len(labels_main)):
                                random_indices_main = random.sample([idx for idx in range(class_num()) if idx != labels_main[t]], 9) 
                                main_selected_outputs[t, random_indices_main] = main_outputs[t, random_indices_main]
                                main_selected_outputs[t, labels_main[t]] = main_outputs[t, labels_main[t]]

                                random_indices_unit = [relationship[str(idx)] for idx in random_indices_main]
                                for j, indices in enumerate(random_indices_unit):
                                    unit_selected_outputs[t, indices] = unit_outputs[t, indices]

                        for t in range(main_outputs.shape[0]):
                            for main_selected_class, main_selected_output in enumerate(main_outputs[t]):
                                unit_selected_classes = relationship[str(main_selected_class)]
                                for unit_class in unit_selected_classes:
                                    main_tmp[t, unit_class] = main_selected_output.item()

                        comb_outputs = a * unit_selected_outputs + b * main_tmp

                    main_score, main_top_idx = torch.sort(main_outputs.detach(), dim=-1, descending=True)
                    unit_score, unit_top_idx = torch.sort(unit_outputs.detach(), dim=-1, descending=True)
                    comb_score, comb_top_idx = torch.sort(comb_outputs.detach(), dim=-1, descending=True)

                    main_top10_score = softmax(main_score, dim=-1)[:, :10]
                    unit_top10_score = softmax(unit_score, dim=-1)[:, :10]
                    comb_top10_score = softmax(comb_score, dim=-1)[:, :10]

                    main_top10 = main_top_idx[:, :10]
                    unit_top10 = unit_top_idx[:, :10]
                    comb_top10 = comb_top_idx[:, :10]


                    step, step_preds = step_test(inputs, device)


                    for idx in range(len(inputs)):
                        # main_result = {int(main_top10[idx][i]): float(main_top10_score[idx][i]) for i in range(10)}
                        # unit_result = {int(unit_top10[idx][i]): float(unit_top10_score[idx][i]) for i in range(10)}
                        comb_result = {int(comb_top10[idx][i]): float(comb_top10_score[idx][i]) for i in range(10)}
                        comb_result["step"] = {int(step[i]): float(step_preds[i]) for i in range(len(step))}
                        #append gt_name to end of comb_result
                        comb_result['gt_name'] = gt_name[idx]
                        # Save to individual files
                        # main_file_path = f"/mnt/hdd/dbstjswo505/workspace/hyundae_samho/jyshim/Ship/dictionary/result_dict/{camid}.json"
                        # unit_file_path = f"/mnt/hdd/dbstjswo505/workspace/hyundae_samho/jyshim/Ship/dictionary/result_dict/{camid}.json"
                        comb_file_path = f"/home/admin/workspace/bppm/Ship/dictionary/result_dict/{camid[idx]}.json"

                        # with open(main_file_path, 'w') as file:
                        #     json.dump(main_result, file)
                        # with open(unit_file_path, 'w') as file:
                        #     json.dump(unit_result, file)
                        with open(comb_file_path, 'w') as file:
                            json.dump(comb_result, file)


            en_results = ensemble(result_path)



