import json
import pandas as pd
from collections import defaultdict
import glob
import os
from copy import deepcopy
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='consistency')
    parser.add_argument(
        '--pred_data', type=str, required=True,
        help='CSV file containing model predictions')
    parser.add_argument(
        '--gold_data', type=str, required=True,
        help='path to gold data')
    parser.add_argument(
        '--gpt_pred', type=str, default=None,
        help='path to GPT preds')
    parser.add_argument(
        '--human', type=str, required=True,
        help='path to save predictions for human analysis')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    data = pd.read_csv(args.pred_data)
    gold_data = pd.read_csv(args.gold_data)
    #gpt_test = json.load(open(args.gpt_pred, "r"))

    # accuracy
    acc = []
    pred_responses = []
    gold_labels = []
    for j in range(len(data)):
        if "output" in data.columns:
            ct_gold_label = eval(data["output"][j])["output"]
        else:
            ct_gold_label = eval(data["label"][j])["output"]
        gold_labels.append(ct_gold_label)
        if "pred" in data.columns:
            p = data["pred"][j]
        else:
            p = data["pred_0"][j]
        if "llama3" in args.pred_data and "instruct" not in args.pred_data:
            if "Response:" not in p:
                acc.append(0)
                pred_responses.append(p)
                continue
            ct_response = p[p.index("Response:"):]
            ct_response = ct_response.replace("Response: ", "").lstrip().rstrip()
        else:
            ct_response = p
        if "{" in ct_response and "}" in ct_response:
            ind1 = ct_response.index("{")
            if "instruct" in args.pred_data:
                ind2 = ct_response.rindex("}")
            else:
                ind2 = ct_response.index("}")
        else:
            acc.append(0)
            pred_responses.append(ct_response)
            continue
        ct_response1 = ct_response[ind1:ind2+1]
        pred_responses.append(ct_response1)
        try:
            ct_dict = eval(ct_response1)
            ct_op = ct_dict["output"]
        except:
            ct_op = "none"

        if ct_op == ct_gold_label:
            acc.append(1)
        else:
            acc.append(0)
    print("Accuracy: ", sum(acc)*100/len(acc))
    assert len(gold_labels) == len(pred_responses)

    # # checking if gold and predicted data are in same order
    # for i in range(len(data)):
    #     assert data["label"][i] == gold_data["output"][i]
    #     assert gold_data["input"][i] in gpt_test

    # checking faithfullness
    faithfullness = []
    faithfullness_score = []
    consistency_acc = []
    consistency1 = []
    consistency2 = []
    for i in range(len(gold_labels)):
        p = pred_responses[i]
        g = gold_labels[i]
        try:
            pred_dict=eval(p)
            p1 = pred_dict["output"]
            count_predicted_label = {"YES":0, "NO": 0, "MAYBE": 0}

            for key in pred_dict.keys():
                if key in ["final score", "output"]:
                    continue
                if "YES" in pred_dict[key]:
                    count_predicted_label["YES"] += 1
                elif "NO" in pred_dict[key]:
                    count_predicted_label["NO"] += 1
                elif "MAYBE" in pred_dict[key]:
                    count_predicted_label["MAYBE"] += 1

            condition_yes = count_predicted_label["YES"] + count_predicted_label["MAYBE"] > count_predicted_label["NO"]
            condition_no = count_predicted_label["NO"] + count_predicted_label["MAYBE"] > count_predicted_label["YES"]

            if condition_yes and p1 == "YES":
                f1 = 1
            elif condition_no and p1 == "NO":
                f1 = 1
            else:
                f1 = 0

            if pred_dict["final score"] >= 0.5 and p1 == "YES":
                f2 = 1
            elif pred_dict["final score"] <= 0.5 and p1 == "NO":
                f2 = 1
            else:
                f2 = 0

            if f1 and f2 and (p1 == g):
                consistency_acc.append(1)
            else:
                consistency_acc.append(0)

            if f1 and (p1 == g):
                consistency1.append(1)
            else:
                consistency1.append(0)

            if f2 and (p1 == g):
                consistency2.append(1)
            else:
                consistency2.append(0)

            faithfullness.append(f1)
            faithfullness_score.append(f2)

        except:
            consistency_acc.append(0)
            faithfullness.append(0)
            faithfullness_score.append(0)
            consistency1.append(0)
            consistency2.append(0)
            pass

    faithfullness = np.array(faithfullness)
    faithfullness_score = np.array(faithfullness_score)
    average_consistency = (faithfullness + faithfullness_score)/2
    consistency_acc = np.array(consistency_acc)
    consistency1 = np.array(consistency1)
    consistency2 = np.array(consistency2)
    print("Consistency with intermediate labels: ", np.mean(faithfullness), len(faithfullness))
    print("Consistency with final score: ", np.mean(faithfullness_score), len(faithfullness_score))
    print("Overall consistency: ", np.mean(average_consistency), len(average_consistency))

    # save for in-house human analysis
    save_dict = {"Input Texts": [], "Property + Output": []}
    save_dict1 = {"Number": [], "Analyze": []}
    label0, label1 = 0, 0
    ind = 0
    for i in range(len(data)):
        ct_question = gold_data["input"][i]
        ct_gold_label = gold_labels[i]
        ct_op = pred_responses[i].replace(', "', ',\n"').replace("{ ", "{\n").replace("}", "\n}")
        try:
            ct_pred_dict = eval(ct_op)
            if ct_pred_dict["output"] != ct_gold_label: continue
            if ct_gold_label=="NO":
                if label0 < 25:
                    label0 += 1
                else:
                    continue
            if ct_gold_label=="YES":
                if label1 < 25:
                    label1 += 1
                else:
                    continue
        except:
            continue
        save_dict["Input Texts"].append(ct_question + "\n\nGOLD LABEL: " + ct_gold_label)
        save_dict1["Analyze"].append(ct_question + "\n\nGOLD LABEL: " + ct_gold_label)
        save_dict1["Number"].append(ind)
        save_dict["Input Texts"].extend(["  "]*(len(ct_pred_dict.keys())-1))
        for key in ct_pred_dict.keys():
            save_dict["Property + Output"].append(key + ": " + str(ct_pred_dict[key]))
            save_dict1["Analyze"].append(key + ": " + str(ct_pred_dict[key]))
            save_dict1["Number"].append(ind)
        save_dict1["Analyze"].append("--------------------------------------------------------------------------------")
        save_dict1["Number"].append(ind)
        ind += 1
    save_df = pd.DataFrame.from_dict(save_dict1)
    save_df.to_csv(args.human)



if __name__=="__main__":
    main()
