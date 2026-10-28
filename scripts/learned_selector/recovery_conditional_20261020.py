"""Single factorized-onset contrast. No action/map authority or new scale fit."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .recovery_parameters_20261019 import LOG_BOUND


class ConditionalRecoveryHead(nn.Module):
    def __init__(self, old_head):
        super().__init__()
        hidden=old_head.scale.in_features
        self.active=nn.Linear(hidden,1).to(dtype=old_head.scale.weight.dtype)
        self.age=nn.Linear(hidden,25).to(dtype=old_head.scale.weight.dtype)
        self.scale=copy.deepcopy(old_head.scale).requires_grad_(False)

    def forward(self, features):
        active=self.active(features).squeeze(-1);age=self.age(features)
        joint=torch.cat((F.logsigmoid(-active)[...,None],
                         F.logsigmoid(active)[...,None]+age.log_softmax(-1)),dim=-1)
        raw=self.scale(features)
        mean=LOG_BOUND*raw[...,:4].tanh();sigma=.001+.199*raw[...,4:].sigmoid()
        return dict(active_logits=active,age_logits=age,onset_logits=joint,
                    log_scale_mean=mean,log_scale_sigma=sigma,correction=torch.exp(-mean))


def conditional_loss(output,onset,log_scale,eligible,scale_mask):
    active=eligible&(onset!=0);mask=eligible[...,None]&scale_mask
    if not eligible.any() or not mask.any():raise ValueError('no eligible observations')
    binary=F.binary_cross_entropy_with_logits(output['active_logits'][eligible],(onset[eligible]!=0).double())
    age=F.cross_entropy(output['age_logits'][active],onset[active]-1) if active.any() else output['age_logits'].sum()*0.
    mean=output['log_scale_mean'][mask];sigma=output['log_scale_sigma'][mask]
    scale=(.5*((log_scale[mask]-mean)/sigma).square()+sigma.log()).mean()
    return binary+age+scale,dict(binary=binary,conditional_age=age,frozen_scale_nll=scale)


def onset_metrics(probabilities,onset,eligible,episode):
    """Both binary marginal and 26-class joint denominators; never action counts."""
    active_prob=1-probabilities[:,0];active_pred=active_prob>=.5
    age=probabilities[:,1:].argmax(-1)+1;pred=probabilities.argmax(-1)
    truth=onset!=0;recent=eligible&(onset>0)&(onset<25)
    recent_age=recent&(age<25)
    located=recent_age&active_pred
    nominal=eligible&~truth;positive=eligible&truth
    error=(age-onset).double()*.25  # errors between bin centres, not exact timestamp
    mean=lambda x:float(x.mean()) if x.numel() else None
    return dict(events=len(onset),eligible=int(eligible.sum()),unknown_ineligible=int((~eligible).sum()),
        binary_threshold=.5,binary_threshold_is_offline_not_commit=True,
        nominal_events=int(nominal.sum()),active_events=int(positive.sum()),
        binary_false_positive_frames=int((nominal&active_pred).sum()),binary_false_negative_frames=int((positive&~active_pred).sum()),
        binary_brier=mean((active_prob[eligible]-truth[eligible].double()).square()),
        joint_confusion=torch.bincount(onset[eligible]*26+pred[eligible],minlength=676).reshape(26,26).tolist(),
        conditional_age_confusion=torch.bincount((onset[positive]-1)*25+age[positive]-1,minlength=625).reshape(25,25).tolist(),
        recent_truth=int(recent.sum()),recent_truth_binary_missed=int((recent&~active_pred).sum()),
        recent_truth_conditional_predicted_older=int((recent&(age==25)).sum()),
        recent_conditionally_localized=int(recent_age.sum()),recent_binary_and_localized=int(located.sum()),
        recent_conditional_bincentre_mae_sec=mean(error[recent_age].abs()),
        recent_conditional_bincentre_rmse_sec=float(error[recent_age].square().mean().sqrt()) if recent_age.any() else None,
        recent_binary_and_localized_mae_sec=mean(error[located].abs()),
        older_truth=int((eligible&(onset==25)).sum()),older_predicted_recent=int((eligible&(onset==25)&(age<25)).sum()),
        joint_recent_prediction_frames=int((eligible&(pred>0)&(pred<25)).sum()),
        episodes=[dict(episode=int(i),all_events=int((episode==i).sum()),active_truth=int(((episode==i)&truth).sum()),
            active_eligible=int(((episode==i)&positive).sum()),active_binary_detected=int(((episode==i)&positive&active_pred).sum()),
            active_unknown=int(((episode==i)&truth&~eligible).sum()),
            recent_truth=int(((episode==i)&recent).sum()),recent_binary_and_localized=int(((episode==i)&located).sum()))
            for i in torch.unique(episode) if i>0],
        scale_direction_fix_claim=False,map_trajectory_gain='NOT_TESTED',public_output=False)
