import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets  # assuming this supports batched anchors

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets, anchors):
        """
        Multi-task loss (objectness + classification + localization) with hard negative mining.
        """
        device = predictions[0].device

        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0

        batch_size = len(targets)

        for scale_idx, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            num_anchors = C // (5 + self.num_classes)

            # Reshape predictions: [B, num_anchors, 5 + num_classes, H, W]
            pred = pred.view(B, num_anchors, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 5+num_classes]
            pred = pred.view(B, -1, 5 + self.num_classes)     # [B, num_anchors*H*W, 5+num_classes]

            scale_anchors = anchors[scale_idx].to(device)     # shape: [num_anchors*H*W, 4]
            # DO NOT repeat!

            pred_obj = pred[..., 4] # [B, N]
            pred_cls = pred[..., 5:] # [B, N, num_classes]
            pred_loc = pred[..., :4] # [B, N, 4]

            for b in range(B):
                t_boxes = targets[b]["boxes"].to(device)
                t_labels = targets[b]["labels"].to(device)

                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    scale_anchors, t_boxes, t_labels
                )

                # Ensure masks are 1D and on correct device
                pos_mask = pos_mask.to(device)
                neg_mask = neg_mask.to(device)

                obj_targets = pos_mask.float()
                pred_obj_b = pred_obj[b]
                if pred_obj_b.shape != obj_targets.shape:
                    raise ValueError(f"Shape mismatch: pred_obj {pred_obj_b.shape} vs obj_targets {obj_targets.shape}")
                
                # (rest unchanged)
                loss_obj = F.binary_cross_entropy_with_logits(pred_obj_b, obj_targets, reduction="none")

                # Classification & Localization (only for positives)
                if pos_mask.sum() > 0:
                    pred_cls_pos = pred_cls[b][pos_mask]
                    labels_pos = matched_labels[pos_mask]
                    pred_loc_pos = pred_loc[b][pos_mask]
                    boxes_pos = matched_boxes[pos_mask]

                    loss_cls = F.cross_entropy(pred_cls_pos, labels_pos, reduction="mean")
                    loss_loc = F.smooth_l1_loss(pred_loc_pos, boxes_pos, reduction="mean")
                else:
                    loss_cls = torch.tensor(0.0, device=device)
                    loss_loc = torch.tensor(0.0, device=device)

                # Hard negative mining (3:1)
                num_pos = pos_mask.sum().item()
                if num_pos > 0:
                    num_neg = min(3 * num_pos, neg_mask.sum().item())
                    if num_neg > 0:
                        neg_loss_vals = loss_obj[neg_mask]
                        _, neg_idx = torch.topk(neg_loss_vals, num_neg)
                        selected_neg_mask = torch.zeros_like(neg_mask)
                        selected_neg_mask[neg_mask.nonzero(as_tuple=True)[0][neg_idx]] = True
                    else:
                        selected_neg_mask = torch.zeros_like(neg_mask)
                else:
                    selected_neg_mask = torch.zeros_like(neg_mask)

                # Final objectness loss (positives + selected negatives)
                loss_obj_final = (loss_obj[pos_mask].sum() + loss_obj[selected_neg_mask].sum()) / max(1, num_pos)

                total_obj_loss += loss_obj_final
                total_cls_loss += loss_cls
                total_loc_loss += loss_loc

        # Average over batch
        loss_dict = {
            "loss_obj": total_obj_loss / batch_size,
            "loss_cls": total_cls_loss / batch_size,
            "loss_loc": total_loc_loss / batch_size,
        }
        loss_dict["loss_total"] = loss_dict["loss_obj"] + loss_dict["loss_cls"] + loss_dict["loss_loc"]

        return loss_dict

        

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.
        
        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio
            
        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = pos_mask.sum().item()
        num_neg = min(int(ratio * num_pos), neg_mask.sum().item())
        if num_neg == 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)
        neg_loss = loss[neg_mask]
        _, idx = torch.topk(neg_loss, num_neg)
        selected_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        selected_neg_mask[neg_mask.nonzero(as_tuple=True)[0][idx]] = True
        return selected_neg_mask

