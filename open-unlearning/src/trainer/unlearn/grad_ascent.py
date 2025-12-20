from trainer.unlearn.base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss

    def compute_loss_per_token_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss_per_token = self.compute_causal_lm_loss_per_token(outputs.logits, forget_inputs['labels'])
        loss = self.calculate_superloss(loss_per_token)
        loss = -loss
        return (loss, outputs) if return_outputs else loss

    def compute_loss_per_sample_superloss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        loss_per_sample = self.compute_causal_lm_loss_per_sample(outputs.logits, forget_inputs['labels'])
        loss = self.calculate_superloss(loss_per_sample)
        loss = -loss
        return (loss, outputs) if return_outputs else loss
