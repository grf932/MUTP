import os
import copy
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from dataset_type import dataset_type_dict

def get_image(data_path, dataset_type):
    image_dict = {}
    train_type_0 = dataset_type_dict[dataset_type[0]]
    train_type_1 = dataset_type_dict[dataset_type[1]]
    path_list = [
        os.path.join(data_path, 'train', train_type_0),
        os.path.join(data_path, 'train', train_type_1),
        os.path.join(data_path, 'test', train_type_0),
        os.path.join(data_path, 'test', train_type_1),
    ]
    for path in path_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + '/'+ filename).convert('RGB')
                im = data_transforms(im)
                image_dict[filename.split('/')[-1].split('.')[0]] = im
            except:
                print("error", filename)
    print("image length " + str(len(image_dict)))
    return image_dict

def count(labels):
    type_1_num, type_0_num = 0, 0
    for label in labels:
        if label == 0:
            type_0_num += 1
        elif label == 1:
            type_1_num += 1
    return type_1_num, type_0_num

def get_data(data_path, mode, dataset_type, image_dict):
    file = data_path+mode+'_'+dataset_type+'_data.csv'
    data = pd.read_csv(file, sep=',')
    text = data['text'].tolist()
    image_url = data['image_url'].tolist()
    label = data['label'].tolist()

    texts, images, labels = [], [], []
    ct = 0
    for url in image_url:
        image_name = url.split('/')[-1].split('.')[0]
        if image_name in image_dict:
            texts.append(text[ct].split())
            images.append(image_dict[image_name])
            labels.append(label[ct])
        ct += 1
    print('trafficnet:', len(texts), 'samples...')

    type_1_num, type_0_num = count(labels)

    type_0 = dataset_type_dict[dataset_type[0]]
    type_1 = dataset_type_dict[dataset_type[1]]
    print(f"{mode} contains: {type_1_num} {type_1}, {type_0_num} {type_0}.")

    rt_data = {'text': texts, 'image': images, 'label': labels}
    return rt_data

def get_vocab(train_data, test_data):
    vocab = defaultdict(float)
    all_text = train_data['text'] + test_data['text']
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text

def add_unknown_words(w2v, vocab, min_df=1, k=32):
    added_words_count = 0
    for word in vocab:
        if word not in w2v.wv.key_to_index and vocab[word] >= min_df:
            w2v.wv.add_vector(word, np.random.uniform(-0.25, 0.25, k))
            added_words_count += 1
    print(f"Total number of words added: {added_words_count}")

def get_W(w2v, k=32):
    word_idx_map = dict()
    W = np.zeros(shape=(len(w2v.wv.key_to_index) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in w2v.wv.key_to_index:
        W[i] = w2v.wv.get_vector(word)
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def word2vec(text, word_idx_map, seq_len):
    word_embedding = []
    for sentence in text:
        sentence_embed = []
        for i, word in enumerate(sentence):
            sentence_embed.append(word_idx_map[word])
        while len(sentence_embed) <= seq_len:
            sentence_embed.append(0)
        sentence_embed = sentence_embed[:seq_len]
        word_embedding.append(copy.deepcopy(sentence_embed))
    return word_embedding

def load_data(args):
    print('Loading image...')
    image_dict = get_image(args.data_path, args.dataset_type)

    print('Loading data...')
    train_data = get_data(args.data_path, 'train', args.dataset_type, image_dict)
    test_data = get_data(args.data_path, 'test', args.dataset_type, image_dict)

    vocab, all_text = get_vocab(train_data, test_data)
    print("vocab size: " + str(len(vocab)))
    max_len = len(max(all_text, key=len))
    print("max sentence length: " + str(max_len))
    avg_len = sum(len(sentence) for sentence in all_text) / len(all_text)
    print("average sentence length: " + str(avg_len))

    print("Loading word2vec...")
    word_embedding_path = args.data_path + 'w2v.pickle'
    w2v = pickle.load(open(word_embedding_path, 'rb'))
    print("Number of words already in word2vec:" + str(len(w2v.wv.key_to_index)))
    add_unknown_words(w2v, vocab, k=32)
    W, word_idx_map = get_W(w2v, k=32)

    num_words = W.shape[0]
    print(f"Number of words in the word embedding matrix: {num_words}")

    args.vocab_size = len(vocab)
    print('Translate text to embedding...')
    train_data['text_embed'] = word2vec(train_data['text'], word_idx_map, args.seq_len)
    test_data['text_embed'] = word2vec(test_data['text'], word_idx_map, args.seq_len)

    text_embed = train_data['text_embed']+test_data['text_embed']
    image = train_data['image']+test_data['image']
    label = train_data['label']+test_data['label']

    return np.array(text_embed), np.array(image), np.array(label), W
