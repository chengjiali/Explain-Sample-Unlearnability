import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.utils import compute_undial_loss
from trainer.unlearn.grad_diff import GradDiff


class UNDIAL(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = compute_undial_loss(
            model, self.ref_model, forget_inputs, self.beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def compute_loss_per_token_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_loss_per_token, forget_outputs = self.compute_undial_loss_per_token(
            model, self.ref_model, forget_inputs, self.beta
        )
        forget_loss = self.calculate_superloss(forget_loss_per_token)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss_per_token = self.compute_retain_loss_per_token(model=model, retain_inputs=retain_inputs)
        retain_loss = self.calculate_superloss(retain_loss_per_token)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def compute_loss_per_sample_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_loss_per_token, forget_outputs = self.compute_undial_loss_per_token(
            model, self.ref_model, forget_inputs, self.beta
        )
        forget_loss_per_sample = self._convert_per_token_loss_to_per_sample_loss(forget_loss_per_token, forget_inputs['labels'])
        forget_loss = self.calculate_superloss(forget_loss_per_sample)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss_per_sample = self.compute_retain_loss_per_sample(model=model, retain_inputs=retain_inputs)
        retain_loss = self.calculate_superloss(retain_loss_per_sample)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
    
    def compute_undial_loss_per_token(self, model, ref_model, inputs, beta):
        # Forward pass on the student (trainable) model
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        # Forward pass on the teacher model (no grad)
        with torch.no_grad():
            teacher_logits = ref_model(**inputs).logits
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

        # Build the mask that identifies the tokens need to be unlearned
        mask = torch.zeros_like(shift_teacher_logits)
        batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
        seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
        mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

        # Adjust teacher logits: subtract di_strength on the correct token
        pre_softmax = shift_teacher_logits - mask * beta
        soft_label = F.softmax(pre_softmax, dim=-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            soft_label.view(-1, soft_label.size(-1)),
        )
        return loss, outputs
    
    def compute_undial_loss_per_sample(self, model, ref_model, inputs, beta):
        loss_per_token, outputs = self.compute_undial_loss_per_token(model, ref_model, inputs, beta)
        loss_per_sample = self._convert_per_token_loss_to_per_sample_loss(loss_per_token, inputs['labels'])

        return loss_per_sample, outputs