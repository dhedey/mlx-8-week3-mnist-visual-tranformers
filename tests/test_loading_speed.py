from model.composite_dataset import CompositeDataset, BesCombine, UltraFastBesCombine
from torch.utils.data import DataLoader
from model.composite_dataset import sequence_collate_fn
import time
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops


#
#
#
torch.manual_seed(47)
random.seed(47)




if __name__ == '__main__':
    num_images = 10000
    batch_size = 512
    num_workers = 4

    # dataset = CompositeDataset(length=num_images)

    # start_time = time.time()
    # for i in range(num_images):
    #     dataset[i] #load 100000 images
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds to load {num_images} images")


    # dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=sequence_collate_fn, num_workers=num_workers, )

    # start_time = time.time()
    # for batch in dataloader:
    #     pass
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds to load {num_images} images with DataLoader with batch size {batch_size} and {num_workers} workers.")

    #UltraFastBesCombine
    start_time = time.time()
    ultra_fast_combine_dataset = UltraFastBesCombine(train=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds to setup UltraFastBesCombine")
    start_time = time.time()
    for i in range(len(ultra_fast_combine_dataset)):
        ultra_fast_combine_dataset[i]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds to load {len(ultra_fast_combine_dataset)} images")

    #UltraFastBesCombine with DataLoader
    ultra_fast_combine_dataloader =  DataLoader(ultra_fast_combine_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    start_time = time.time()
    for batch in ultra_fast_combine_dataloader:
        pass
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds to load {len(ultra_fast_combine_dataset)} images with DataLoader with batch size {batch_size} and {num_workers} workers.")

    #Do the same thing with the Combine dataset
    combine_dataset = BesCombine(train=True)
    start_time = time.time()
    for i in range(len(combine_dataset)):
        combine_dataset[i]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds to load {len(combine_dataset)} images")

    combine_dataloader =  DataLoader(combine_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    start_time = time.time()
    for batch in combine_dataloader:
        pass
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds to load {len(combine_dataset)} images with DataLoader with batch size {batch_size} and {num_workers} workers.")

#
#
#
