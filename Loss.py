import numpy
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    def __init__(self, exp_ratio = 5):
        super().__init__()
        self.exp_ratio = exp_ratio
        self.mode = 1

    def forward(self, x, y, label):
        """
        :param x: [batch, d_model] often peptide
        :param y: [batch, d_model] often chrom
        :param label: [batch]
        :return:
        """
        if self.mode == 0:
            B, _ = x.shape
            self.exp_ratio = np.log(B)
            device = x.device
            x_t = x[label == 1]
            x_f = x[label == 0]
            y_t = y[label == 1]
            y_f = y[label == 0]
            B_t, _ = x_t.shape
            B_f, _ = x_f.shape

            sim_t_vec = F.cosine_similarity(x_t, y_t, dim=-1) * self.exp_ratio #[B_t]
            sim_all_vec= F.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1).reshape(-1) * self.exp_ratio #[B*B]

            ratio = torch.sum(torch.exp(sim_t_vec), dim=-1)/torch.sum(torch.exp(sim_all_vec), dim=-1)
            loss = -1 * torch.log(ratio)
        elif self.mode == 1:
            B, _ = x.shape
            self.exp_ratio = np.log(B)
            device = x.device
            x_t = x[label == 1]
            x_f = x[label == 0]
            y_t = y[label == 1]
            y_f = y[label == 0]
            B_t, _ = x_t.shape
            B_f, _ = x_f.shape

            sim_t_vec = F.cosine_similarity(x_t, y_t, dim=-1) * self.exp_ratio  # [B_t]
            sim_all_vec = F.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1).reshape(
                -1) * self.exp_ratio  # [B*B]

            ratio = torch.sum(torch.exp(sim_t_vec), dim=-1) / torch.sum(torch.exp(sim_all_vec), dim=-1)
            loss = -1 * torch.log(ratio)
        return loss, torch.tensor(0), torch.tensor(0)

class RINCE(nn.Module):
    def __init__(self, lamb=0.5, q = 0.5, exp_ratio = 5):
        super().__init__()
        self.lamb = lamb
        self.q = q
        self.exp_ratio = exp_ratio

    def forward(self, x, y, label):
        """
        :param x: [batch, d_model] often peptide
        :param y: [batch, d_model] often chrom
        :param label: [batch]
        :return:
        """
        B, _ = x.shape
        self.exp_ratio = np.log(B)
        device = x.device
        x_t = x[label == 1]
        x_f = x[label == 0]
        y_t = y[label == 1]
        y_f = y[label == 0]
        B_t, _ = x_t.shape
        B_f, _ = x_f.shape

        sim_t_vec = F.cosine_similarity(x_t, y_t, dim=-1) * self.exp_ratio #[B_t]
        sim_all_vec= F.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1).reshape(-1) * self.exp_ratio #[B*B]

        loss = -1 * torch.sum(torch.exp(self.q * sim_t_vec), dim=-1)/self.q + torch.pow(self.lamb * torch.sum(torch.exp(sim_all_vec), dim=-1), exponent=self.q)/self.q
        return loss, torch.tensor(0), torch.tensor(0)

class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            alpha = 0.25,
            delta = 0.1,
            beta = 0.01,
            gamma =0.01,
            is_dist = True,
            only_true = False,
            loss_mode = 0,
            consider_inner_dist = True
    ):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.only_true = only_true
        self.is_dist = is_dist
        self.loss_mode = loss_mode
        self.consider_inner_dist = consider_inner_dist

    def forward(self, peptide_feature, chrom_feature, label):
        """
        :param x: [batch, d_model] often peptide
        :param y: [batch, d_model] often chrom
        :param label: [batch]
        :return:
        """
        B, d_model = peptide_feature.shape
        device = peptide_feature.device
        peptide_feature_t = peptide_feature[label == 1]
        peptide_feature_f = peptide_feature[label == 0]
        chrom_feature_t = chrom_feature[label == 1]
        chrom_feature_f = chrom_feature[label == 0]
        B_t, _ = peptide_feature_t.shape
        B_f, _ = peptide_feature_f.shape

        if self.is_dist:
            # true part
            target = torch.arange(0, B, 1).to(device)[label == 1]
            loss_t_fc = nn.CrossEntropyLoss()
            logits = -1 * torch.cdist(chrom_feature_t, peptide_feature, p=2)  # cos similarity [batch_t, batch]
            loss_t = loss_t_fc(logits, target)
            if self.loss_mode == 1:
                ap_loss_t = torch.mean(torch.norm( chrom_feature_t - peptide_feature_t , dim=-1, p=2, keepdim=False) , dim=-1)
                loss_t = self.beta * ap_loss_t + (1-self.beta) * loss_t

            # false part
            softmax_fc = nn.Softmax(dim=-1)
            KL_fc = nn.KLDivLoss(reduction='batchmean')
            equal_prob = (torch.ones(B_f, B) / B).to(device)
            dist_matrix = -1 * torch.cdist(peptide_feature_f, chrom_feature, p=2)
            prob = softmax_fc(dist_matrix)
            loss_f_1 = KL_fc(prob.log(), equal_prob)
            dist_matrix = -1 * torch.cdist(chrom_feature_f, peptide_feature, p=2)
            prob = softmax_fc(dist_matrix)
            loss_f_2 = KL_fc(prob.log(), equal_prob)
            loss_f = (loss_f_1 + loss_f_2) / 2
            if self.loss_mode == 1:
                ap_loss_f = torch.mean(-1 * torch.norm( chrom_feature_f - peptide_feature_f , dim=-1, p=2, keepdim=False), dim=-1)
                loss_f = self.gamma * ap_loss_f + (1-self.gamma) * loss_f

            if self.only_true:
                loss = loss_t
            else:
                loss = (1 - self.alpha) * loss_t + self.alpha * loss_f
            
            #inner dist part
            if self.consider_inner_dist:
                loss_t_fc = nn.CrossEntropyLoss()
                target = torch.arange(0, B, 1).to(device)
                logits = -1 * torch.cdist(chrom_feature, chrom_feature, p=2) / (d_model**0.5)
                inner_dist_loss = loss_t_fc(logits, target)
                logits = -1 * torch.cdist(peptide_feature, peptide_feature, p=2) / (d_model**0.5)
                inner_dist_loss = 0.5 * inner_dist_loss + 0.5 * loss_t_fc(logits, target)

                loss = (1- self.delta) * loss + self.delta * inner_dist_loss

        else:
            exp_ratio = np.log(B)
            #true part
            target = torch.arange(0, B, 1).to(device)[label == 1]
            loss_t_fc = nn.CrossEntropyLoss()
            logits = F.cosine_similarity(chrom_feature_t[:, None, :], peptide_feature[None, :, :], dim=-1) * exp_ratio  # cos similarity [batch_t, batch]
            loss_t = loss_t_fc(logits, target)

            #false part
            softmax_fc = nn.Softmax(dim=-1)
            KL_fc = nn.KLDivLoss(reduction='batchmean')
            equal_prob = (torch.ones(B_f, B) / B).to(device)
            sim_matrix = F.cosine_similarity(peptide_feature_f[:, None, :], chrom_feature[None, :, :], dim=-1) * exp_ratio
            prob = softmax_fc(sim_matrix)
            loss_f_1 = KL_fc(prob.log(), equal_prob)
            sim_matrix = F.cosine_similarity(chrom_feature_f[:, None, :], peptide_feature[None, :, :], dim=-1) * exp_ratio
            prob = softmax_fc(sim_matrix)
            loss_f_2 = KL_fc(prob.log(), equal_prob)
            loss_f = (loss_f_1 + loss_f_2) / 2

            if self.only_true:
                loss = loss_t
            else:
                loss = (1 - self.alpha) *loss_t + self.alpha * loss_f

            #inner dist part
            if self.consider_inner_dist:
                loss_t_fc = nn.CrossEntropyLoss()
                target = torch.arange(0, B, 1).to(device)
                logits = F.cosine_similarity(chrom_feature[:, None, :], chrom_feature[None, :, :], dim=-1) * exp_ratio
                inner_dist_loss = loss_t_fc(logits, target)
                logits = F.cosine_similarity(peptide_feature[:, None, :], peptide_feature[None, :, :], dim=-1) * exp_ratio
                inner_dist_loss = 0.5 * inner_dist_loss + 0.5 * loss_t_fc(logits, target)
                loss = (1- self.delta) * loss + self.delta * inner_dist_loss

        if self.consider_inner_dist:
            return loss, loss_t, loss_f, inner_dist_loss
        else:
            return loss, loss_t, loss_f





if __name__ == "__main__":
    import torch
    B = 64
    x = torch.randn(3, 10)
    y = torch.randn(5, 10)

    print(x)
    print(y)
    print(torch.cdist(x, y, p=2))


