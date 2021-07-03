import torch
from torch import optim
from torch import nn
import numpy as np
import mstcn
import pickle, time
import random
from sklearn import metrics
import copy


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx,\
           test_labels_80, test_num_each_80, test_start_vidx

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

train_labels_80, train_num_each_80, train_start_vidx,\
    val_labels_80, val_num_each_80, val_start_vidx,\
    test_labels_80, test_num_each_80, test_start_vidx = get_data('./train_val_paths_labels1.pkl')




with open("./LFB/g_LFB50_train.pkl", 'rb') as f:
    g_LFB_train = pickle.load(f)

with open("./LFB/g_LFB50_val.pkl", 'rb') as f:
    g_LFB_val = pickle.load(f)
with open("./LFB/g_LFB50_test.pkl", 'rb') as f:
    g_LFB_test = pickle.load(f)

print("load completed")

print("g_LFB_train shape:", g_LFB_train.shape)
print("g_LFB_val shape:", g_LFB_val.shape)
print("g_LFB_test shape:", g_LFB_test.shape)

out_features = 7
num_workers = 3
batch_size = 1
features_per_seconds = 25
features_subsampling = 1
mstcn_causal_conv = True
learning_rate = 5e-4
min_epochs = 4
max_epochs = 8
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim= 2048
mstcn_stages = 2

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

weights_train = np.asarray([1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,])
criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
#criterion_phase = nn.CrossEntropyLoss()

model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_accuracy_phase = 0.0
correspond_train_acc_phase = 0.0
best_epoch = 0

train_we_use_start_idx_80 = [x for x in range(40)]
val_we_use_start_idx_80 = [x for x in range(8)]
test_we_use_start_idx_80 = [x for x in range(32)]
#random.shuffle(train_we_use_start_idx_80)
for epoch in range(max_epochs):
    torch.cuda.empty_cache()
    train_idx_80 = []
    model.train()
    train_loss_phase = 0.0
    train_corrects_phase = 0
    batch_progress = 0.0
    running_loss_phase = 0.0
    minibatch_correct_phase = 0.0
    train_start_time = time.time()
    for i in train_we_use_start_idx_80:
        optimizer.zero_grad()
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each_80[i]):
            labels_phase.append(train_labels_80[j][0])
        labels_phase = torch.LongTensor(labels_phase)
        if use_gpu:
            labels_phase = labels_phase.to(device)
        else:
            labels_phase = labels_phase
        #print(labels_phase.size())
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = (torch.Tensor(long_feature)).to(device)
        video_fe = long_feature.transpose(2, 1)
        #print(long_feature.size())

        # inputs = inputs.view(-1, 3, 224, 224)
        # emb = resnet.forward(inputs).detach()
        # diff = (emb-long_feature[:,-1]).data.cpu().numpy()
        # print(np.sum(np.abs(diff)))
        y_classes = model.forward(video_fe)
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):  ### make the interuption free stronge the more layers.
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = criterion_phase(p_classes, labels_phase)
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)

        _, preds_phase = torch.max(y_classes[stages-1].squeeze().transpose(1, 0).data, 1)

        loss = clc_loss
        #print(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

        running_loss_phase += clc_loss.data.item()
        train_loss_phase += clc_loss.data.item()

        batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
        train_corrects_phase += batch_corrects_phase
        minibatch_correct_phase += batch_corrects_phase



    train_elapsed_time = time.time() - train_start_time
    train_accuracy_phase = float(train_corrects_phase) / len(train_labels_80)
    train_average_loss_phase = train_loss_phase

    # Sets the module in evaluation mode.
    model.eval()
    val_loss_phase = 0.0
    val_corrects_phase = 0
    val_start_time = time.time()
    val_progress = 0
    val_all_preds_phase = []
    val_all_labels_phase = []
    val_acc_each_video = []

    with torch.no_grad():
        for i in val_we_use_start_idx_80:
            labels_phase = []
            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                labels_phase.append(val_labels_80[j][0])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=val_start_vidx[i],
                                            lfb=g_LFB_val, LFB_length=val_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_classes = model.forward(video_fe)
            stages = y_classes.shape[0]
            clc_loss = 0
            for j in range(stages):  ### make the interuption free stronge the more layers.
                p_classes = y_classes[j].squeeze().transpose(1, 0)
                ce_loss = criterion_phase(p_classes, labels_phase)
                clc_loss += ce_loss
            clc_loss = clc_loss / (stages * 1.0)

            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
            p_classes = y_classes[-1].squeeze().transpose(1, 0)
            loss_phase = criterion_phase(p_classes, labels_phase)

            val_loss_phase += loss_phase.data.item()

            val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/val_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))


    val_elapsed_time = time.time() - val_start_time
    val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
    val_acc_video = np.mean(val_acc_each_video)
    val_average_loss_phase = val_loss_phase

    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
    val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    test_start_time = time.time()

    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            labels_phase = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][0])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_classes = model.forward(video_fe)
            stages = y_classes.shape[0]
            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
            p_classes = y_classes[-1].squeeze().transpose(1, 0)

            test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

    test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time

    print('epoch: {:4d}'
          ' train in: {:2.0f}m{:2.0f}s'
          ' train loss(phase): {:4.4f}'
          ' train accu(phase): {:.4f}'
          ' valid in: {:2.0f}m{:2.0f}s'
          ' valid loss(phase): {:4.4f}'
          ' valid accu(phase): {:.4f}'
          ' valid accu(video): {:.4f}'
          ' test in: {:2.0f}m{:2.0f}s'
          ' test accu(phase): {:.4f}'
          ' test accu(video): {:.4f}'
          .format(epoch,
                  train_elapsed_time // 60,
                  train_elapsed_time % 60,
                  train_average_loss_phase,
                  train_accuracy_phase,
                  val_elapsed_time // 60,
                  val_elapsed_time % 60,
                  val_average_loss_phase,
                  val_accuracy_phase,
                  val_acc_video,
                  test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_accuracy_phase,
                  test_acc_video))

    print("val_precision_each_phase:", val_precision_each_phase)
    print("val_recall_each_phase:", val_recall_each_phase)
    print("val_precision_phase", val_precision_phase)
    print("val_recall_phase", val_recall_phase)
    print("val_jaccard_phase", val_jaccard_phase)

    if val_accuracy_phase > best_val_accuracy_phase:
        best_val_accuracy_phase = val_accuracy_phase
        best_test_accuracy_phase = test_accuracy_phase
        correspond_train_acc_phase = train_accuracy_phase
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = epoch

    save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
    save_test_phase = int("{:4.0f}".format(best_test_accuracy_phase * 10000))
    save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
    base_name = "TeCNO50" \
                + "_epoch_" + str(best_epoch) \
                + "_train_" + str(save_train_phase) \
                + "_val_" + str(save_val_phase) \
                + "_test_" + str(save_test_phase)
    #torch.save(best_model_wts, "./best_model/TeCNO/" + base_name + ".pth")
    print("best_epoch", str(best_epoch))


