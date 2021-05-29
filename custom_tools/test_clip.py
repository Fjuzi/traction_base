#from slowfast.datasets import loader
import torch
from custom_tools.kinetics_clip import Kinetics
import clip
import numpy as np
from PIL import Image
import os

shuffle = False
drop_last = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



true_class_correspondences = open("./custom_tools/FULL.csv","r")
all_lines = true_class_correspondences.readlines()
scent_classnames = []
scent_classvalue = []
phrases = []
for line in all_lines:
    scent_classnames.append(line.split(",")[2])
    scent_classvalue.append(int(line.split(",")[3]))
    phrases.append(line.split(",")[5][:-2])

text = clip.tokenize(phrases).to(device)


dataset = Kinetics(preprocess, "/data/Peter/Data/Smells/")

loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=None,
        )

prev_class = ""
f = open("dumb.txt","w")
_prev_unique_video_idx = ""
for cur_iter, (inputs, labels, video_idx, meta, _unique_video_idx, _temporal_sample_index, _spatial_sample_index, _video_name) in enumerate(loader):
#for cur_iter, (inputs, labels, video_idx, meta, _unique_video_idx) in enumerate(loader):
    #print(str(cur_iter))
    scene_inputs = inputs[:,:,32,:,:]
    inputs = torch.squeeze(scene_inputs)
    video_idx = torch.squeeze(video_idx)
    _unique_video_idx = torch.squeeze(_unique_video_idx)

    with torch.no_grad():
        image_features = model.encode_image(inputs)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #values, indices = similarity[0].topk(5)
    #print(phrases[indices[0]] + ";" + phrases[indices[1]] + ";" + phrases[indices[2]] + ";" + phrases[indices[3]]+ ";" + phrases[indices[4]])
    #values, indices = similarity[1].topk(5)
    #values, indices = similarity[2].topk(5)
    #values, indices = similarity[3].topk(5)
    #values, indices = similarity[4].topk(5)

    '''if prev_class != labels[0]:
        print(labels[0])
        prev_class = labels[0]
        f.close()
        index = true_classnames.index(labels[0])
        filenamesm = "./results_all2/" + str(index) + ".txt" 
        if os.path.exists(filenamesm):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        f = open(filenamesm, append_write)'''
    
    #values, indices = similarity[0].topk(600)
    #n_values = values.cpu().detach().numpy()
    #n_indicies = indices.cpu().detach().numpy()
    #f.write(np.array2string( n_indicies, separator=',' ) + ";")

    video_idx = video_idx.cpu()
    labels = labels.cpu()

    video_idx_numpy = video_idx.numpy()
    #preds_numpy = preds.numpy()
    labels_numpy = labels.numpy()
    unique_video_idx = _unique_video_idx.numpy()
    temporal_sample_index = _temporal_sample_index.numpy()
    spatial_sample_index = _spatial_sample_index.numpy()
    video_name = _video_name

    for i in range(video_idx_numpy.shape[0]):
        if str(labels_numpy[i]) != _prev_unique_video_idx:
            _prev_unique_video_idx = str(labels_numpy[i])
            f.close()
            filenamesm = "./results/resuls_clip/" + _prev_unique_video_idx +".txt"
            if os.path.exists(filenamesm):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
                #print(labels_numpy[i])
            f = open(filenamesm, append_write)
        f.write(str(video_name[i]))
        f.write(";")
        f.write(str(unique_video_idx[i]))
        f.write(";")
        f.write(str(temporal_sample_index[i]))
        f.write(";")
        f.write(str(spatial_sample_index[i]))
        f.write(";")
        f.write(str(labels_numpy[i]))
        f.write(";")
        n_similarity = similarity[i].cpu().detach().numpy()
        tmp2 = list(n_similarity)
        for element in tmp2:
            f.write("%1.8f,"%element)
        f.write("\n")

    '''for local_i in range(5):
        n_similarity = similarity[local_i].cpu().detach().numpy()
        f.write(str(_unique_video_idx.item()) + ";")
        f.write(np.array2string( n_similarity, separator=',', formatter = {'float_kind':lambda x: "%.8f"%x}) + ";")
        index = true_classnames.index(labels[0])
        f.write(str(index) + "\n")'''

    '''print(cur_iter)
    print(inputs.size())
    a = inputs.numpy()
    tmp = (a - np.min(a))/(np.max(a)-np.min(a))
    tmp = tmp*255
    tmp = tmp.astype(np.uint8)
    im = Image.fromarray(tmp[0,:,:,:])
    im.save("00.jpeg")
    asd'''

f.close()