import numpy as np
import torchvision
import torch

def get_torchvision_data(name, root="../data/", download=True,
						transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
					    			#torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
									]),
						target_transform=None, **kwargs):
	data = {}

	for split in ["train","test"]:
		raw_data = getattr(torchvision.datasets,name)(root=root, train=split=="train", transform=transform, target_transform=target_transform, download=download)
		
		#for var in ["x","y"]:
		data[split+"_x"] = torch.stack([app[0] for app in raw_data])
		data[split+"_y"] = torch.tensor([app[1] for app in raw_data])

	return data