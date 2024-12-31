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

class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            t_ratio = 0.75,
            inner_dist_ratio = 0.1,
            norm_ratio = 0.01,
            is_dist = True,
    ):
        super().__init__()
        self.t_ratio = t_ratio
        self.norm_ratio = norm_ratio
        self.inner_dist_ratio = inner_dist_ratio
        self.is_dist = is_dist

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=-1)

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

        # Compute true part loss
        target_true = torch.arange(0, B, 1).to(device)[label == 1]
        dist_matrix_true = self.compute_distance_matrix(chrom_feature_t, peptide_feature)    # distance [batch_t, batch]
        ce_loss_true = self.cross_entropy_loss(dist_matrix_true, target_true)
        norm_loss_true = torch.mean(torch.norm(chrom_feature_t - peptide_feature_t, dim=-1, p=2), dim=-1)
        loss_true = self.norm_ratio * norm_loss_true + (1 - self.norm_ratio) * ce_loss_true

        # false part
        equal_prob = (torch.ones(B_f, B) / B).to(device)
        dist_matrix_false_1 = self.compute_distance_matrix(peptide_feature_f, chrom_feature)
        prob_1 = self.softmax(dist_matrix_false_1)
        loss_false_1 = self.kl_div_loss(prob_1.log(), equal_prob)

        dist_matrix_false_2 = self.compute_distance_matrix(chrom_feature_f, peptide_feature)
        prob_2 = self.softmax(dist_matrix_false_2)
        loss_false_2 = self.kl_div_loss(prob_2.log(), equal_prob)

        loss_false = (loss_false_1 + loss_false_2) / 2
        norm_loss_false = torch.mean(-torch.norm(chrom_feature_f - peptide_feature_f, dim=-1, p=2), dim=-1)
        loss_false = self.norm_ratio * norm_loss_false + (1 - self.norm_ratio) * loss_false

        loss = self.t_ratio * loss_true + (1 - self.t_ratio) * loss_false
        inner_loss = self.compute_inner_distance_loss(chrom_feature, peptide_feature)
        loss = self.inner_dist_ratio * inner_loss + (1 - self.inner_dist_ratio) * loss

        return {'loss': loss, 'loss_t': loss_true, 'loss_f': loss_false, 'inner_loss': inner_loss}

    def compute_inner_distance_loss(self, chrom_feature, peptide_feature):
        B, d_model = peptide_feature.shape
        device = chrom_feature.device
        # Compute inner distance loss for chrom_feature and peptide_feature
        target = torch.arange(0, B, 1).to(device)

        # Compute distance matrices (scaled)
        chrom_dist_matrix = self.compute_distance_matrix(chrom_feature, chrom_feature)
        peptide_dist_matrix = self.compute_distance_matrix(peptide_feature, peptide_feature)

        # Compute CrossEntropy loss for both distance matrices
        chrom_loss = self.cross_entropy_loss(chrom_dist_matrix, target)
        peptide_loss = self.cross_entropy_loss(peptide_dist_matrix, target)

        # Average the two losses
        inner_dist_loss = 0.5 * chrom_loss + 0.5 * peptide_loss

        return inner_dist_loss

    def compute_distance_matrix(self, chrom_feature, peptide_feature):
        B, d_model = peptide_feature.shape
        device = chrom_feature.device
        if self.is_dist:
            dist_matrix = - torch.cdist(chrom_feature, peptide_feature, p=2)
        else:
            exp_ratio = np.log(B)
            dist_matrix = F.cosine_similarity(chrom_feature[:, None, :], peptide_feature[None, :, :], dim=-1) * exp_ratio
        return dist_matrix



if __name__ == "__main__":
    import torch
    B = 64
    x = torch.randn(3, 10)
    y = torch.randn(5, 10)

    print(x)
    print(y)
    print(torch.cdist(x, y, p=2))


