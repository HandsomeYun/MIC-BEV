import pickle
from pprint import pformat

# with open('/home/handsomeyun/BEVFormer/data/v2x_inf/v2xset_infos_temporal_train.pkl', 'rb') as f:
with open('/home/yun/MIC-BEV_Official/data/v2x_4cam_map_weather/v2xset_infos_temporal_all.pkl', 'rb') as f:
    data = pickle.load(f)

#==================FIND THE FIRST SAMPLE==========================================

# Get first entry from 'infos' list
print(len(data['infos']))
first_sample = data['infos'][-1]
# first_sample = data[0]

# Format the dictionary as a string
formatted = pformat(first_sample)

path = '/home/yun/MIC-BEV_Official/third_sample.txt'
# Save to a text file
with open(path, 'w') as out_file:
    out_file.write(formatted)

print(f"✅ Third sample saved to {path}")

#=====================USE TARGET TOKEN TO FIND=====================================

# # The sample token to search
# target_token = 'bridgeentry_town07_med_test_fov110_tp300_c_d2_day_s55_-2_000207'

# # Search for the sample with that token
# target_sample = None
# for sample in data['infos']:
#     if sample['token'] == target_token:
#         target_sample = sample
#         break

# if target_sample:
#     # Format and save the result
#     formatted = pformat(target_sample)
#     with open('/home/handsomeyun/BEVFormer/target_sample.txt', 'w') as out_file:
#         out_file.write(formatted)
#     print("✅ Target sample saved to 'target_sample.txt'")
# else:
#     print("❌ Sample not found.")
