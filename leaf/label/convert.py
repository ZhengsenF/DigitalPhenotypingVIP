import json
with open('Soybean Plant (5).json') as fptr:
    file = json.load(fptr)
print(file)
for each_file_name in file['_via_img_metadata']:
    for each_region_index, each_region in enumerate(file['_via_img_metadata'][each_file_name]['regions']):
        file['_via_img_metadata'][each_file_name]['regions'][each_region_index]['shape_attributes']['all_points_x'].pop(-1)
        file['_via_img_metadata'][each_file_name]['regions'][each_region_index]['shape_attributes']['all_points_y'].pop(-1)
        file['_via_img_metadata'][each_file_name]['regions'][each_region_index]['shape_attributes']['name'] = 'polygon'


print(file)
with open('soybean_mi.json', 'w') as fptr:
    json.dump(file, fptr)
