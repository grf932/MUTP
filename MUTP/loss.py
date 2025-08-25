import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-5
T1 = 0.05
T2 = 0.05

class UnsupervisedContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=T1):
        super(UnsupervisedContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())

    def forward(self, q, k):
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)   # (bs,)
        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk/self.temperature), dim=1)

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq/self.temperature), dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, batch_size, num_type_1, num_type_0, device='cuda', temperature=T2):
        super(SupervisedContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.num_type_1 = num_type_1
        self.num_type_0 = num_type_0

        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("type_1_mask", (~torch.eye(num_type_1, num_type_1, dtype=torch.bool).to(device)).float())
        self.register_buffer("type_0_mask", (~torch.eye(num_type_0, num_type_0, dtype=torch.bool).to(device)).float())

    def compute_loss(self, feature, label):
        index_1 = torch.nonzero(label).squeeze()
        index_0 = torch.nonzero(label == 0).squeeze()
        ft_1 = torch.index_select(feature, dim=0, index=index_1)
        ft_0 = torch.index_select(feature, dim=0, index=index_0)

        similarity_matrix_11 = F.cosine_similarity(ft_1.unsqueeze(1), ft_1.unsqueeze(0), dim=2)
        similarity_matrix_00 = F.cosine_similarity(ft_0.unsqueeze(1), ft_0.unsqueeze(0), dim=2)
        similarity_matrix_10 = F.cosine_similarity(ft_1.unsqueeze(1), ft_0.unsqueeze(0), dim=2)
        similarity_matrix_01 = F.cosine_similarity(ft_0.unsqueeze(1), ft_1.unsqueeze(0), dim=2)

        nominator_1 = torch.sum(self.type_1_mask * torch.exp(similarity_matrix_11 / self.temperature), dim=1)
        nominator_0 = torch.sum(self.type_0_mask * torch.exp(similarity_matrix_00 / self.temperature), dim=1)

        denominator_1 = nominator_1 + torch.sum(torch.torch.exp(similarity_matrix_10 / self.temperature), dim=1)
        denominator_0 = nominator_0 + torch.sum(torch.torch.exp(similarity_matrix_01 / self.temperature), dim=1)

        loss_1 = torch.sum(-torch.log(nominator_1 / denominator_1 + eps)) / self.num_type_1
        loss_0 = torch.sum(-torch.log(nominator_0 / denominator_0 + eps)) / self.num_type_0
        loss = loss_1 + loss_0
        return loss

    def forward(self, text, image, label):
        text = F.normalize(text, dim=1)
        image = F.normalize(image, dim=1)
        loss = self.compute_loss(text, label) + self.compute_loss(image, label)
        return loss

class MultimodalInteractionLoss(nn.Module):
    def __init__(self, pad_id=0):
        super(MultimodalInteractionLoss, self).__init__()
        self.pad_id = pad_id

    def forward(self, logits, text):
        # Reshape logits to (batch_size, num_classes, seq_len)
        logits = logits.permute(0, 2, 1)
        text = text.long()
        # Compute cross-entropy loss, ignoring pad_id
        mim_loss = F.cross_entropy(logits, text, ignore_index=self.pad_id)

        return mim_loss

