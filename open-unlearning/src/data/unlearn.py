import os
import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

        self.selected_indices = list(range(len(forget)))    # Trainer will create a data loader before training starts.
                                                            # Later on epoch starts, this will be updated by self.curriculum()

    def curriculum(self, curr_epoch, num_train_epochs, cl_method, sample_difficulty):
        # Curriculum configuration
        num_chunks = num_train_epochs // 2  # 5 chunks for 10 epochs
        if cl_method == 'hard_to_easy':
            descending = True
        elif cl_method == 'easy_to_hard':
            descending=False
        else:
            raise

        # Current training epoch
        curr_epoch = int(curr_epoch)    # curr_epoch can be float
        max_stage = min(num_chunks, (curr_epoch // 2) + 1)

        # Sort by difficulty and create curriculum chunks
        sorted_indices = torch.argsort(sample_difficulty, descending=descending)
        chunks = torch.chunk(sorted_indices, num_chunks)
        self.selected_indices = torch.cat(chunks[:max_stage]).tolist()
        if int(os.environ.get('RANK')) == 0:
            print(f"[Curriculum]: Epoch {curr_epoch} | Using {max_stage}/{num_chunks} chunks | {len(self.selected_indices)}/{len(sorted_indices)} samples")

    def curriculum_step_based(self, curr_step, num_train_steps, cl_method, sample_difficulty):
        # Curriculum configuration
        num_chunks = num_train_steps // 16  # 5 chunks for 80 steps
        if cl_method == 'hard_to_easy':
            descending = True
        elif cl_method == 'easy_to_hard':
            descending=False
        else:
            raise

        # Current training epoch
        chunk_size = num_train_steps / num_chunks
        curr_stage = int(curr_step // chunk_size)
        curr_stage = min(num_chunks, curr_stage + 1)

        # Sort by difficulty and create curriculum chunks
        sorted_indices = torch.argsort(sample_difficulty, descending=descending)
        chunks = torch.chunk(sorted_indices, num_chunks)
        self.selected_indices = torch.cat(chunks[: curr_stage]).tolist()
        if int(os.environ.get('RANK')) == 0:
            print(f"[Curriculum]: Step {curr_stage} | Using {curr_stage}/{num_chunks} chunks | {len(self.selected_indices)}/{len(sorted_indices)} samples")

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len([self.forget[i] for i in self.selected_indices])
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[(retain_idx+rank) % len(self.retain)]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item
