import torch

from utils.dataloader import gc_dataset, collate_fn
from baselines.baselines import baselines

previous_day = 7
source_root = r"./data/"

if __name__ == "__main__":
    dataset = "gc"

    dataset_train = gc_dataset(source_root, dataset=dataset, data_type="train", previous_day=previous_day)
    kwds_train = {"shuffle": True, "num_workers": 0, "batch_size": 1}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)

    dataset_validation = gc_dataset(source_root, dataset=dataset, data_type="validation", previous_day=previous_day)
    kwds_validation = {"shuffle": False, "num_workers": 0, "batch_size": 1}
    val_loader = torch.utils.data.DataLoader(dataset_validation, collate_fn=collate_fn, **kwds_validation)

    dataset_test = gc_dataset(source_root, dataset=dataset, data_type="test", previous_day=previous_day)
    kwds_test = {"shuffle": False, "num_workers": 0, "batch_size": 1}
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds_test)

    baselines(train_loader, val_loader, test_loader)
    
    # persistent forecast the best, slightly better than markov + persistent
