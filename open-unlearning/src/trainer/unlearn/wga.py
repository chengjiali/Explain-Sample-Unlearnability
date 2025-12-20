import torch.nn as nn
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss


class WGA(GradDiff):
    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_wga_loss(
            model=model, inputs=forget_inputs, beta=self.beta
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.gamma * forget_loss + self.alpha * retain_loss
        )  # default gamma=1.0 alpha=1.0
        return (loss, forget_outputs) if return_outputs else loss

    def compute_loss_per_token_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss_per_token, forget_outputs = self.compute_wga_loss_per_token(
            model=model, inputs=forget_inputs, beta=self.beta
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

        loss = (
            self.gamma * forget_loss + self.alpha * retain_loss
        )  # default gamma=1.0 alpha=1.0
        return (loss, forget_outputs) if return_outputs else loss

    def compute_loss_per_sample_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss_per_sample, forget_outputs = self.compute_wga_loss_per_sample(
            model=model, inputs=forget_inputs, beta=self.beta
        )
        forget_loss = self.calculate_superloss(forget_loss_per_sample)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss_per_sample = self.compute_retain_loss_per_sample(model=model, retain_inputs=retain_inputs)
        retain_loss = self.calculate_superloss(retain_loss_per_sample)

        loss = (
            self.gamma * forget_loss + self.alpha * retain_loss
        )  # default gamma=1.0 alpha=1.0
        return (loss, forget_outputs) if return_outputs else loss

    def compute_wga_loss_per_token(self, model, inputs, beta, remove_ignore_index=True):
        outputs = model(**inputs)
        labels = inputs["labels"]
        labels = labels.to(outputs.logits.device)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        weight_ce = ((-lm_loss).exp().detach()) ** beta
        
        if remove_ignore_index:
            forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100]#.mean() # Original WGA loss returns mean loss
        else:
            forget_loss = -(weight_ce * lm_loss)
        return forget_loss, outputs

    def compute_wga_loss_per_sample(self, model, inputs, beta):
        loss_per_token, outputs = self.compute_wga_loss_per_token(model, inputs, beta, remove_ignore_index=False)
        loss_per_sample = self._convert_per_token_loss_to_per_sample_loss(loss_per_token, inputs['labels'])

        return loss_per_sample, outputs
