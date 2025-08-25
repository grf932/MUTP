import argparse
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from arguments import parse_arguments
from data_loader import *
from datasets import MyDataset
from model import TraffiCFUS
from loss import SupervisedContrastiveLoss, UnsupervisedContrastiveLoss, MultimodalInteractionLoss
import json
from dataset_type import dataset_type_dict

warnings.filterwarnings("ignore")

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
    type_1_num, type_0_num = 0, 0
    for label in labels:
        if label == 0:
            type_0_num += 1
        elif label == 1:
            type_1_num += 1
    return type_1_num, type_0_num

def shuffle_dataset(text, image, label):
    assert len(text) == len(image) == len(label)
    rp = np.random.permutation(len(text))
    text = text[rp]
    image = image[rp]
    label = label[rp]

    return text, image, label


def save_results(args, results, fold=None):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if fold is not None:
        output_file = os.path.join(args.output_path, f"results_fold_{fold}_{time.strftime('%Y%m%d-%H%M')}.json")
    else:
        output_file = os.path.join(args.output_path, f"results_average_{time.strftime('%Y%m%d-%H%M')}.json")

    results_dict = {
        'seed': args.seed,
        'dataset_type': args.dataset_type,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'num_epoch': args.num_epoch,
        'remarks': args.remarks,
        'results': results,
        'k': fold if fold is not None else "average"
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

def main(args):
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
    valid_type_0_pre_sum, valid_type_0_recall_sum, valid_type_0_f1_sum = 0., 0., 0.
    valid_type_1_pre_sum, valid_type_1_recall_sum, valid_type_1_f1_sum = 0., 0., 0.

    train, valid = {}, {}

    type_0 = dataset_type_dict[args.dataset_type[0]]
    type_1 = dataset_type_dict[args.dataset_type[1]]
    one_name, zero_name = type_1, type_0

    for i in range(K):
        print('-' * 25, 'Fold:', i + 1, '-' * 25)
        train['text'], train['image'], train['label'], valid['text'], valid['image'], valid['label'] = \
            get_kfold_data(K, i, text, image, label)

        train_loader = DataLoader(dataset=MyDataset(train), batch_size=args.batch_size, shuffle=False)
        valid_loader = DataLoader(dataset=MyDataset(valid), batch_size=args.batch_size, shuffle=False)

        print('Building model...')

        model = TraffiCFUS(W, args.vocab_size, args.d_text, args.seq_len, args.img_size, args.patch_size, args.d_model,
                           args.num_filter, args.num_class, args.num_layer, args.dropout)
        model.to(device)

        if torch.cuda.is_available():
            print("CUDA")
            model.cuda()

        criterion_dsf = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1 = 0., 0., 0., 0.
        best_valid_0_pre, best_valid_0_recall, best_valid_0_f1 = 0., 0., 0.
        best_valid_1_pre, best_valid_1_recall, best_valid_1_f1 = 0., 0., 0.

        loss_list = []
        acc_list = []
        for epoch in range(args.num_epoch):
            train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
            start_time = time.time()
            cls_loss = []

            model.train()
            for j, (train_text, train_image, train_labels) in enumerate(train_loader):
                num_r, num_nr = count(train_labels)
                train_text, train_image, train_labels = to_var(train_text), to_var(train_image), to_var(train_labels)

                criterion_supcon = SupervisedContrastiveLoss(batch_size=train_text.shape[0], num_type_1=num_r, num_type_0=num_nr)
                criterion_unscon = UnsupervisedContrastiveLoss(batch_size=train_text.shape[0])
                criterion_mim = MultimodalInteractionLoss()
                optimizer.zero_grad()

                text_outputs, image_outputs, label_outputs, _, logits= model(train_text, train_image)

                loss_dsf = criterion_dsf(label_outputs, train_labels.long())
                loss_sup = criterion_supcon(text_outputs, image_outputs, train_labels.long())
                loss_uns = criterion_unscon(text_outputs, image_outputs)
                loss_mim = criterion_mim(logits, train_text)

                train_loss = loss_dsf + args.alpha * loss_sup + args.beta * loss_uns + args.gamma * loss_mim
                train_loss.backward()
                optimizer.step()
                pred = torch.max(label_outputs, 1)[1]
                train_accuracy = torch.eq(train_labels, pred.squeeze()).float().mean()
                train_losses.append(train_loss.item())
                train_acc.append(train_accuracy.item())
                cls_loss.append(loss_dsf.item())

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

                    _, _, label_outputs, _, features = model(valid_text, valid_image, args.need_tsne_data)
                    label_outputs = F.softmax(label_outputs, dim=1)
                    pred = torch.max(label_outputs, 1)[1]
                    if j == 0:
                        valid_pred = to_np(pred.squeeze())
                        valid_y = to_np(valid_labels.squeeze())
                    else:
                        valid_pred = np.concatenate((valid_pred, to_np(pred.squeeze())), axis=0)
                        valid_y = np.concatenate((valid_y, to_np(valid_labels.squeeze())), axis=0)

                    if args.need_tsne_data:
                        save_path = os.path.join(args.output_path, f"K{i}/epoch{epoch}/batch{j}")
                        os.makedirs(save_path, exist_ok=True)
                        np.save(os.path.join(save_path, f"features.npy"), features)
                        np.save(os.path.join(save_path, f"labels.npy"), to_np(valid_labels))

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
                target_names = [type_0, type_1]
                report = metrics.classification_report(valid_y, valid_pred, output_dict=True, target_names=target_names)
                type_0_report = report[type_0]
                best_valid_0_pre = type_0_report['precision']
                best_valid_0_recall = type_0_report['recall']
                best_valid_0_f1 = type_0_report['f1-score']
                type_1_report = report[type_1]
                best_valid_1_pre = type_1_report['precision']
                best_valid_1_recall = type_1_report['recall']
                best_valid_1_f1 = type_1_report['f1-score']

        valid_acc_sum += best_valid_acc
        valid_pre_sum += best_valid_pre
        valid_recall_sum += best_valid_recall
        valid_f1_sum += best_valid_f1
        print('best_valid_acc:{:.6f}, best_valid_pre:{:.6f}, best_valid_recall:{:.6f}, best_valid_f1:{:.6f}'.
              format(best_valid_acc, best_valid_pre, best_valid_recall, best_valid_f1))
        valid_type_0_pre_sum += best_valid_0_pre
        valid_type_0_recall_sum += best_valid_0_recall
        valid_type_0_f1_sum += best_valid_0_f1
        valid_type_1_pre_sum += best_valid_1_pre
        valid_type_1_recall_sum += best_valid_1_recall
        valid_type_1_f1_sum += best_valid_1_f1

        # Collect results for the current fold
        results = {
            'accuracy': best_valid_acc,
            'f1': best_valid_f1,
            'one_name': one_name,
            'one_precision': best_valid_1_pre,
            'one_recall': best_valid_1_recall,
            'one_f1': best_valid_1_f1,
            'zero_name': zero_name,
            'zero_precision': best_valid_0_pre,
            'zero_recall': best_valid_0_recall,
            'zero_f1': best_valid_0_f1
        }

        # Save results for the current fold
        save_results(args, results, fold=i+1)

    print('=' * 40)
    print('Accuracy:{:.5f}, F1:{:.5f}'.format(valid_acc_sum / K, valid_f1_sum / K))


    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        one_name, valid_type_1_pre_sum / K, one_name, valid_type_1_recall_sum / K, one_name, valid_type_1_f1_sum / K))
    print('{} Precision:{:.5f}, {} Recall:{:.5f}, {} F1:{:.5f}'.format(
        zero_name, valid_type_0_pre_sum / K, zero_name, valid_type_0_recall_sum / K, zero_name, valid_type_0_f1_sum / K))

    # Collect average results
    results = {
        'accuracy': valid_acc_sum / K,
        'f1': valid_f1_sum / K,
        'one_name': one_name,
        'one_precision': valid_type_1_pre_sum / K,
        'one_recall': valid_type_1_recall_sum / K,
        'one_f1': valid_type_1_f1_sum / K,
        'zero_name': zero_name,
        'zero_precision': valid_type_0_pre_sum / K,
        'zero_recall': valid_type_0_recall_sum / K,
        'zero_f1': valid_type_0_f1_sum / K
    }

    # Save average results
    save_results(args, results)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()

    main(args)
