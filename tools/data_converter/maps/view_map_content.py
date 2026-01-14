import numpy as np

path = "/home/yun/MIC-BEV_Official/data/map/Town05/-11.50,-125.10/bev_label_map_200.npy"
bev_map = np.load(path)

print("Shape:", bev_map.shape)
print("Data type:", bev_map.dtype)
print("Unique labels:", np.unique(bev_map))
