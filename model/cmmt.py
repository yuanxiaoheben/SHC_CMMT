from .text_embedding import Embedding
from .transformer_encoder import Encoder
from .transformer_decoder import Decoder
from .cm_decoder import V2Tdecoder, T2Vdecoder, VideoSpan, OrderGenerate, LocationRegression, IoURegression
import torch.nn as nn
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler
class CMMT(nn.Module):
    def __init__(self,configs, bert_model=None):
        super(CMMT, self).__init__()
        self.embedding_net = Embedding(configs=configs, bert_model=bert_model)
        self.encoder_text = Encoder(configs)
        self.decoder_text = Decoder(configs)
        self.encoder_video = Encoder(configs)
        self.decoder_video = Decoder(configs)
        self.video_ffn = nn.Linear(configs.video_feature_dim, configs.d_model)
        self.t2v_decoder = T2Vdecoder(configs)
        self.v2t_decoder = V2Tdecoder(configs)
        self.span_out = VideoSpan(configs)
        self.order_out = OrderGenerate(configs)
        self.location = LocationRegression(configs)
        self.iou = IoURegression(configs)
        self.mse_loss = nn.MSELoss()
        self.iou_loss_func = nn.SmoothL1Loss()


    def cmt(self, query_features, video_features): 
        # vfeats (seg_num, batch, len, dim)
        t_processed = []
        for _,query in enumerate(query_features):
            _, query_embed = self.embedding_net(query)
            t_processed.append(query_embed)
        t_processed = torch.stack(t_processed, dim=1)
        t_processed = self.encoder_text(t_processed)
        t_processed = self.decoder_text(t_processed)
        v_encoded = self.encoder_video(self.video_ffn(video_features))
        v_encoded = self.decoder_video(v_encoded)
        t2v_out = self.t2v_decoder(t_processed, v_encoded)
        v2t_out = self.v2t_decoder(v_encoded, t_processed)
        return t2v_out, v2t_out
    
    def forward(self, bert_input, vfeats): 
        t2v_out, v2t_out = self.cmt(bert_input, vfeats)
        order_out, span_out = self.order_out(t2v_out), self.span_out(v2t_out)
        start_reg, end_reg = self.location(t2v_out, v2t_out)
        ious_pred = self.iou(t2v_out, v2t_out)
        return start_reg, end_reg, order_out, span_out, ious_pred
    
    def proposal_loss(self, start_prob, end_prob, true_start, true_end):
        sum_loss = self.mse_loss(start_prob, true_start) + self.mse_loss(end_prob, true_end)
        return sum_loss
    
    def ious_loss(self, pred_iou, start_prob, end_prob, true_start, true_end):
        target_iou = self.segment_tiou(start_prob, end_prob, true_start, true_end)
        loss = self.iou_loss_func(pred_iou, target_iou.detach())
        return loss
    
    def segment_tiou(self, start_prob, end_prob, true_start, true_end):

        # gt: [batch, 1, 2], detections: [batch, 56, 2]
        # calculate interaction
        inter_max_xy = torch.min(end_prob, true_end)
        inter_min_xy = torch.max(start_prob, true_start)
        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

        # calculate union
        union_max_xy = torch.max(end_prob, true_end)
        union_min_xy = torch.min(start_prob, true_start)
        union = torch.clamp((union_max_xy - union_min_xy), min=0)

        iou = inter / (union+1e-6)
        return iou
