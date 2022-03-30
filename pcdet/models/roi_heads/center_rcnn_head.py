import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, loss_utils, box_utils, box_coder_utils

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class CenterRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        pre_channel = self.model_cfg.BEV_FEATURE_EXTRACTOR.IN_CHANNEL * 5
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.iou_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC
        )

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def bev_feature_extractor(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:
            features:
                (B*N, C)
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)

        min_x, min_y = 0, -40
        voxel_size_x, voxel_size_y = 0.05, 0.05
        down_sample_ratio = 4
        features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            num_rois = rois[b_id].size(0)
            corner_2d = box_utils.boxes_to_corners_3d(rois[b_id])[:, :4, :-1]
            center_2d = rois[b_id, :, :2]
            mid_corner_2d = [
                corner_2d[:, [0,1]].mean(dim=1),
                corner_2d[:, [1,2]].mean(dim=1),
                corner_2d[:, [2,3]].mean(dim=1),
                corner_2d[:, [3,0]].mean(dim=1),
                center_2d
            ]
            mid_corner_2d = torch.stack(mid_corner_2d, dim=1)
            mid_corner_2d_relative = (mid_corner_2d - torch.tensor([min_x, min_y]).cuda())/torch.tensor([voxel_size_x, voxel_size_y]).cuda()/down_sample_ratio
            x, y = mid_corner_2d_relative[:, :, 0].flatten(), mid_corner_2d_relative[:, :, 1].flatten()
            feature = bilinear_interpolate_torch(spatial_features_2d[b_id], x, y)
            feature = feature.view(num_rois, -1)
            features_list.append(feature)
        features = torch.cat(features_list, dim=0)
        return features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.bev_feature_extractor(batch_dict)  # (BxN, C)
        batch_size_rcnn = pooled_features.shape[0]

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B*N, 1)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features.squeeze(-1)))

        if not self.training:
            batch_dict['batch_cls_preds'] = rcnn_iou.view(batch_dict['batch_size'], -1, rcnn_iou.shape[-1])
            batch_dict['batch_box_preds'] = batch_dict['rois']
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_iou'] = rcnn_iou
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict



    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(
                rcnn_iou_flat,
                rcnn_iou_labels.float(), reduction='none'
            )
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError

        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)

        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict
