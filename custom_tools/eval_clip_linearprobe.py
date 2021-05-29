import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from tqdm import tqdm
from kinetics_clip import Kinetics


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)



def get_features(dataset, preprocess):
    all_features = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=100, shuffle=True, drop_last = True)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
            
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def calculate_accuracy(l2_lambda_init_idx, l2_lambda_list, fname):
    # Perform logistic regression
    #classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    accs = []
    f = open(fname,"w")
    for idx in l2_lambda_init_idx:
        classifier = LogisticRegression(random_state=0, C=l2_lambda_list[idx], max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
        print(f"Accuracy = {accuracy:.3f}")
        accs.append(accuracy)
        f.write(str(l2_lambda_list[idx]))
        f.write(";")
        f.write(str(accuracy))
        f.write("\n")
    f.close()
    mx_idx =  accs.index(max(accs))
    return l2_lambda_init_idx[mx_idx]

# Calculate the image features
dataset_train = Kinetics(preprocess, train_path)
dataset_test = Kinetics(preprocess, test_path)
train_features, train_labels = get_features(dataset_train, preprocess)
test_features, test_labels = get_features(dataset_test, preprocess)

l2_lambda_list = np.logspace(-3, 0, num=40).tolist()
l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(-1,1, num=7))]

#peak_idx = calculate_accuracy(l2_lambda_init_idx, l2_lambda_list, "res3.txt")
peak_idx = 26
f_sok = open("f_res10.txt","w")
step_span = 8
szamlalo = 5
while step_span > 0:
    left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list)-1)
    peak_idx = calculate_accuracy([left, peak_idx, right], l2_lambda_list, ("res_"  + str(szamlalo)+".txt") )
    print(peak_idx)
    f_sok.write(str(peak_idx))
    step_span //= 2
    szamlalo = szamlalo + 1

print(peak_idx)
f_sok.write(str(peak_idx))
f_sok.close()
'''step_span = 8
lower_idx = max(mx_idx - step_span, 0)
higher_idx = min(mx_idx + step_span, len(l2_lambda_list)-1)
print(calculate_accuracy(np.linspace(lower_idx, higher_idx, higher_idx-lower_idx+1), l2_lambda_list, "res2.txt"))'''

# Evaluate using the logistic regression classifier
'''predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")'''