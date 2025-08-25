"""
An Lao
"""
import argparse
import random
import math
import numpy as np
import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import classification_report
from arguments import parse_arguments
from my_data_loader import *
# from data_loader import *
from datasets import MyDataset
from model import FSRU, FSRU_new, FSRU_new_without_fusion
from loss import FullContrastiveLoss, SelfContrastiveLoss, CaptionLoss
import json
from dataset_type import dataset_type_dict
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import copy

warnings.filterwarnings("ignore")

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def to_var(x):
    if torch.cuda.is_available():
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
    else:
        x = torch.as_tensor(x, dtype=torch.float32)
    return x

def to_np(x):
    return x.data.cpu().numpy()

def get_kfold_data(k, i, text, image ,label):
    fold_size = text.shape[0] // k

    val_start = i * fold_size
    if i != k-1:
        val_end = (i + 1) * fold_size
        text_valid, image_valid, label_valid = text[val_start:val_end], image[val_start:val_end], label[val_start:val_end]
        text_train = np.concatenate((text[0:val_start], text[val_end:]), axis=0)
        image_train = np.concatenate((image[0:val_start], image[val_end:]), axis=0)
        label_train = np.concatenate((label[0:val_start], label[val_end:]), axis=0)
    else:
        text_valid, image_valid, label_valid = text[val_start:], image[val_start:], label[val_start:]
        text_train = text[0:val_start]
        image_train = image[0:val_start]
        label_train = label[0:val_start]

    return text_train, image_train, label_train, text_valid, image_valid, label_valid

def count(labels):
    r, nr = 0, 0
    for label in labels:
        if label == 0:
            nr += 1
        elif label == 1:
            r += 1
    return r, nr

def reshape_transform(tensor):
    # Convert ViT output (B, N, D) to (B, C, H, W) for Grad‑CAM
    tensor = tensor[:, 1:, :]                    # drop [CLS]
    n_patches = tensor.size(1)
    h = w = int(math.sqrt(n_patches))
    tensor = tensor.reshape(tensor.size(0), h, w, -1)
    tensor = tensor.permute(0, 3, 1, 2)          # (B, C, H, W)
    return tensor

def generate_and_save_cam(model, img_tensor, label, save_path):
    """
    Generate Grad‑CAM heat‑map for a single image and save it.

    Args
    ----
    model : torch.nn.Module
        Trained FSRU_new (or its variant).
    img_tensor : torch.Tensor
        3×H×W tensor *after* standard transforms **but before** normalization
        to [-1, 1].  Assumed to be on CPU.
    label : int
        Ground‑truth or predicted class index to guide CAM.
    save_path : str
        PNG path to write.
    """
    # ---------- 1. Locate visual backbone ----------
    backbone = None
    for attr in ("clip_visual", "visual", "vision_encoder", "image_encoder", "backbone"):
        if hasattr(model, attr):
            backbone = getattr(model, attr)
            break
    if backbone is None:
        raise AttributeError(
            "Cannot locate visual backbone in the model. "
            "Specify target_layers manually or extend the attribute list."
        )

    # ---------- 2. Decide target layer(s) & reshape ----------
    use_transform = None
    if hasattr(backbone, "transformer"):  # ViT‑style
        blocks = None
        if hasattr(backbone.transformer, "resblocks"):
            blocks = backbone.transformer.resblocks
        elif hasattr(backbone.transformer, "blocks"):
            blocks = backbone.transformer.blocks
        if blocks and len(blocks) > 0:
            target_layers = [blocks[-1]]
        else:
            target_layers = [backbone]
        use_transform = reshape_transform
    else:  # CNN‑style
        # pick the deepest leaf module (usually conv) as target layer
        leaf_modules = [m for m in backbone.modules() if len(list(m.children())) == 0]
        target_layers = [leaf_modules[-1]] if leaf_modules else [backbone]
        use_transform = None  # no ViT reshape needed

    # ---------- 3. Run Grad‑CAM ----------
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=torch.cuda.is_available(),
        reshape_transform=use_transform,
    )
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(img_tensor.unsqueeze(0), targets=targets)[0]

    # ---------- 4. Overlay & save ----------
    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, heatmap)

def shuffle_dataset(text, image, label):
    assert len(text) == len(image) == len(label)
    rp = np.random.permutation(len(text))
    text = text[rp]
    image = image[rp]
    label = label[rp]

    return text, image, label


def save_results(args, results, fold=None):
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Generate the output file name with date and time
    if fold is not None:
        output_file = os.path.join(args.output_path, f"results_fold_{fold}_{time.strftime('%Y%m%d-%H%M')}.json")
    else:
        output_file = os.path.join(args.output_path, f"results_average_{time.strftime('%Y%m%d-%H%M')}.json")

    # Prepare the results dictionary
    results_dict = {
        'seed': args.seed,
        'dataset_type': args.dataset_type,
        'alpha': args.alpha,
        'beta': args.beta,
        'caption_rate': args.caption_rate,
        'num_epoch': args.num_epoch,
        'remarks': args.remarks,
        'results': results,
        'k': fold if fold is not None else "average"
    }

    # Save the results to the file
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

def main(args):
    # device = torch.device(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading data ...')

    text, image, label, W = load_data(args)
    text, image, label = shuffle_dataset(text, image, label)

    K = args.k
    print('Using K:', K, 'fold cross validation...')

    valid_acc_sum, valid_pre_sum, valid_recall_sum, valid_f1_sum = 0., 0., 0., 0.
    valid_nr_pre_sum, valid_nr_recall_sum, valid_nr_f1_sum = 0., 0., 0.
    valid_r_pre_sum, valid_r_recall_sum, valid_r_f1_sum = 0., 0., 0.

    train, valid = {}, {}

    one_name, zero_name = "Rumor", "Non-Rumor"
    # # 获取dataset_type的两个字符对应的名称
    type_1 = dataset_type_dict[args.dataset_type[0]]
    type_2 = dataset_type_dict[args.dataset_type[1]]
    # 使用这些名称替换one_name和zero_name的赋值
    one_name, zero_name = type_1, type_2

    for i in range(K):
        print('-' * 25, 'Fold:', i + 1, '-' * 25)
        best_model_state = None
        train['text'], train['image'], train['label'], valid['text'], valid['image'], valid['label'] = \
            get_kfold_data(K, i, text, image, label)

        train_loader = DataLoader(dataset=MyDataset(train), batch_size=args.batch_size, shuffle=False)
        valid_loader = DataLoader(dataset=MyDataset(valid), batch_size=args.batch_size, shuffle=False)

        print('Building model...')

        if args.no_fus:
            model = FSRU_new_without_fusion(W, args.vocab_size, args.d_text, args.seq_len, args.img_size, args.patch_size, args.d_model,
                         args.num_filter, args.num_class, args.num_layer, args.dropout,
                         disable_tr=args.no_tr, disable_mim=args.no_mim)
        else:
            model = FSRU_new(W, args.vocab_size, args.d_text, args.seq_len, args.img_size, args.patch_size, args.d_model,
                         args.num_filter, args.num_class, args.num_layer, args.dropout,
                         disable_tr=args.no_tr, disable_mim=args.no_mim)
        model.to(device)

        if torch.cuda.is_available():
            print("CUDA")
            model.cuda()

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1 = 0., 0., 0., 0.
        best_valid_nr_pre, best_valid_nr_f1, best_valid_nr_recall, = 0., 0., 0.
        best_valid_r_pre, best_valid_r_recall, best_valid_r_f1 = 0., 0., 0.

        loss_list = []
        acc_list = []
        for epoch in range(args.num_epoch):
            train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
            start_time = time.time()
            cls_loss = []

            # train
            model.train()
            for j, (train_text, train_image, train_labels) in enumerate(train_loader):
                num_r, num_nr = count(train_labels)
                train_text, train_image, train_labels = to_var(train_text), to_var(train_image), to_var(train_labels)

                # Forward + Backward + Optimize
                criterion_full = FullContrastiveLoss(batch_size=train_text.shape[0], num_r=num_r, num_nr=num_nr)
                criterion_self = SelfContrastiveLoss(batch_size=train_text.shape[0])
                criterion_caption = CaptionLoss()
                optimizer.zero_grad()

                text_outputs, image_outputs, label_outputs, _, logits, text_embed = model(train_text, train_image)

                loss = criterion(label_outputs, train_labels.long())
                loss_full = criterion_full(text_outputs, image_outputs, train_labels.long())
                loss_self = criterion_self(text_outputs, image_outputs)
                if args.no_cl:
                    loss_full = 0.0
                    loss_self = 0.0
                # loss_caption = criterion_caption(logits, train_text)
                loss_caption = 0

                train_loss = loss + args.alpha * loss_full + args.beta * loss_self + loss_caption * args.caption_rate
                # print('Loss:',loss.item(), 'Loss_full:', loss_full.item() * args.alpha, 'Loss_self:', loss_self.item() * args.beta,
                #       'Loss_caption:', loss_caption.item() * args.caption_rate)
                # print('Train_loss:', train_loss.item())
                train_loss.backward()
                optimizer.step()
                pred = torch.max(label_outputs, 1)[1]
                train_accuracy = torch.eq(train_labels, pred.squeeze()).float().mean()  # .sum() / len(train_labels)
                train_losses.append(train_loss.item())
                train_acc.append(train_accuracy.item())
                cls_loss.append(loss.item())

            if epoch % args.decay_step == 0:
                for params in optimizer.param_groups:
                    params['lr'] *= args.decay_rate

            # valid
            model.eval()
            valid_pred, valid_y = [], []
            with torch.no_grad():
                for j, (valid_text, valid_image, valid_labels) in enumerate(valid_loader):
                    valid_text, valid_image, valid_labels = to_var(valid_text), to_var(valid_image), to_var(
                        valid_labels)

                    _, _, label_outputs, _, features, _ = model(valid_text, valid_image, args.need_tsne_data)
                    label_outputs = F.softmax(label_outputs, dim=1)
                    pred = torch.max(label_outputs, 1)[1]
                    if j == 0:
                        valid_pred = to_np(pred.squeeze())
                        valid_y = to_np(valid_labels.squeeze())
                    else:
                        valid_pred = np.concatenate((valid_pred, to_np(pred.squeeze())), axis=0)
                        valid_y = np.concatenate((valid_y, to_np(valid_labels.squeeze())), axis=0)

                    # Save features and labels if need_tsne_data is True
                    if args.need_tsne_data:
                        save_path = os.path.join(args.output_path, f"K{i}/epoch{epoch}/batch{j}")
                        os.makedirs(save_path, exist_ok=True)
                        np.save(os.path.join(save_path, f"features.npy"), features)
                        np.save(os.path.join(save_path, f"labels.npy"), to_np(valid_labels))

            # cur_valid_acc = np.mean(valid_acc)
            cur_valid_acc = metrics.accuracy_score(valid_y, valid_pred)
            valid_pre = metrics.precision_score(valid_y, valid_pred, average='macro')
            valid_recall = metrics.recall_score(valid_y, valid_pred, average='macro')
            valid_f1 = metrics.f1_score(valid_y, valid_pred, average='macro')
            if args.need_tsne_data:
                save_path = os.path.join(args.output_path, f"K{i}/epoch{epoch}")
                os.makedirs(save_path, exist_ok=True)
                # Save validation metrics
                metrics_data = {
                    'accuracy': cur_valid_acc,
                    'precision': valid_pre,
                    'recall': valid_recall,
                    'f1': valid_f1
                }
                with open(os.path.join(save_path, 'validation_metrics.json'), 'w') as json_file:
                    json.dump(metrics_data, json_file, indent=4)
            duration = time.time() - start_time
            print(
                'Epoch[{}/{}], Duration:{:.8f}, Loss:{:.8f}, Train_Accuracy:{:.5f}, Valid_accuracy:{:.5f}'.format(
                    epoch + 1, args.num_epoch, duration, np.mean(train_losses), np.mean(train_acc),
                    cur_valid_acc))
            loss_list.append(np.mean(cls_loss))
            acc_list.append(cur_valid_acc)

            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                best_valid_pre = valid_pre
                best_valid_recall = valid_recall
                best_valid_f1 = valid_f1
                print('Best...')
                # print(metrics.classification_report(valid_y, valid_pred, digits=4))
                target_names = ['non-rumor', 'rumor']
                report = metrics.classification_report(valid_y, valid_pred, output_dict=True, target_names=target_names)
                nr_report = report['non-rumor']
                best_valid_nr_pre = nr_report['precision']
                best_valid_nr_recall = nr_report['recall']
                best_valid_nr_f1 = nr_report['f1-score']
                r_report = report['rumor']
                best_valid_r_pre = r_report['precision']
                best_valid_r_recall = r_report['recall']
                best_valid_r_f1 = r_report['f1-score']
                best_model_state = copy.deepcopy(model.state_dict())

        # ---------- Grad‑CAM generation ----------
        if args.save_cam and best_model_state is not None:
            model.load_state_dict(best_model_state)
            cam_dir = os.path.join(args.output_path, f"K{i+1}", "gradcam")
            os.makedirs(cam_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for idx, (v_text, v_img, v_lbl) in enumerate(valid_loader):
                    v_img_cpu = v_img[0].cpu()
                    generate_and_save_cam(model, v_img_cpu, int(v_lbl[0]),
                                          os.path.join(cam_dir, f"sample_{idx}.png"))
                    if idx >= 9:          # save first 10 samples
                        break

        valid_acc_sum += best_valid_acc
        valid_pre_sum += best_valid_pre
        valid_recall_sum += best_valid_recall
        valid_f1_sum += best_valid_f1
        print('best_valid_acc:{:.6f}, best_valid_pre:{:.6f}, best_valid_recall:{:.6f}, best_valid_f1:{:.6f}'.
              format(best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1))
        valid_nr_pre_sum += best_valid_nr_pre
        valid_nr_recall_sum += best_valid_nr_recall
        valid_nr_f1_sum += best_valid_nr_f1
        valid_r_pre_sum += best_valid_r_pre
        valid_r_recall_sum += best_valid_r_recall
        valid_r_f1_sum += best_valid_r_f1

        # Collect results for the current fold
        results = {
            'accuracy': best_valid_acc,
            'f1': best_valid_f1,
            'one_name': one_name,
            'one_precision': best_valid_r_pre,
            'one_recall': best_valid_r_recall,
            'one_f1': best_valid_r_f1,
            'zero_name': zero_name,
            'zero_precision': best_valid_nr_pre,
            'zero_recall': best_valid_nr_recall,
            'zero_f1': best_valid_nr_f1
        }

        # Save results for the current fold
        save_results(args, results, fold=i+1)

    print('=' * 40)
    print('Accuracy:{:.5f}, F1:{:.5f}'.format(valid_acc_sum / K, valid_f1_sum / K))


    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        one_name, valid_r_pre_sum / K, one_name, valid_r_recall_sum / K, one_name, valid_r_f1_sum / K))
    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        zero_name, valid_nr_pre_sum / K, zero_name, valid_nr_recall_sum / K, zero_name, valid_nr_f1_sum / K))

    # Collect average results
    results = {
        'accuracy': valid_acc_sum / K,
        'f1': valid_f1_sum / K,
        'one_name': one_name,
        'one_precision': valid_r_pre_sum / K,
        'one_recall': valid_r_recall_sum / K,
        'one_f1': valid_r_f1_sum / K,
        'zero_name': zero_name,
        'zero_precision': valid_nr_pre_sum / K,
        'zero_recall': valid_nr_recall_sum / K,
        'zero_f1': valid_nr_f1_sum / K
    }

    # Save average results
    save_results(args, results)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    parser.add_argument('--no_tr', action='store_true', help='disable text refinement module (TR)')
    parser.add_argument('--no_mim', action='store_true', help='disable multimodal interaction module (MIM)')
    parser.add_argument('--no_fus', action='store_true', help='use model without fusion module (FUS)')
    parser.add_argument('--no_cl', action='store_true', help='disable contrastive losses (CL)')
    parser.add_argument('--save_cam', action='store_true',
                        help='Generate Grad‑CAM heat‑maps after each fold')
    args = parser.parse_args()

    if args.no_cl:
        args.alpha = 0.0
        args.beta = 0.0

    main(args)
