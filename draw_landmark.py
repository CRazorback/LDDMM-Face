import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import imageio
import cv2

from lib.utils.transforms import generate_target

# # wflw
# coord = np.array([[14.5955, 29.3104],
#                 [14.8884, 34.9626],
#                 [15.1527, 40.6338],
#                 [15.6358, 46.2863],
#                 [16.3514, 51.8998],
#                 [17.3191, 57.4760],
#                 [18.7827, 62.9341],
#                 [20.7423, 68.2311],
#                 [23.2313, 73.2720],
#                 [26.2557, 77.9831],
#                 [29.7289, 82.3432],
#                 [33.5533, 86.3675],
#                 [37.6665, 90.0613],
#                 [42.0669, 93.3428],
#                 [46.8481, 96.0127],
#                 [52.0945, 97.7077],
#                 [57.6308, 98.1683],
#                 [63.0568, 97.4737],
#                 [68.1323, 95.5779],
#                 [72.7304, 92.7641],
#                 [76.9805, 89.3997],
#                 [80.9604, 85.6560],
#                 [84.6312, 81.5953],
#                 [87.9443, 77.1896],
#                 [90.8033, 72.4645],
#                 [93.1425, 67.4337],
#                 [94.9443, 62.1659],
#                 [96.2695, 56.7463],
#                 [97.1889, 51.2261],
#                 [97.7609, 45.6633],
#                 [98.1354, 40.0820],
#                 [98.3835, 34.4837],
#                 [98.5668, 28.8885],
#                 [25.2569, 23.8813],
#                 [31.2469, 18.7647],
#                 [37.3123, 18.2836],
#                 [43.3783, 18.9954],
#                 [48.8446, 20.4571],
#                 [48.4787, 24.1844],
#                 [42.8093, 23.0315],
#                 [37.0144, 22.3355],
#                 [31.0611, 22.5613],
#                 [64.3343, 20.0850],
#                 [70.5422, 18.5409],
#                 [76.6430, 17.7332],
#                 [82.4168, 18.1771],
#                 [87.8038, 23.0230],
#                 [82.2664, 21.8632],
#                 [76.2433, 21.7613],
#                 [70.2908, 22.5865],
#                 [64.2211, 23.8345],
#                 [56.2833, 30.6741],
#                 [56.3497, 38.2220],
#                 [56.3859, 45.7245],
#                 [56.5613, 52.8416],
#                 [47.4319, 56.4984],
#                 [52.1015, 57.4747],
#                 [56.8513, 58.2663],
#                 [61.2919, 57.3143],
#                 [65.6744, 56.2404],
#                 [30.8884, 31.3181],
#                 [34.3941, 29.1452],
#                 [38.4301, 28.2861],
#                 [42.8379, 29.0884],
#                 [46.3188, 31.8148],
#                 [42.4951, 33.0752],
#                 [38.4345, 33.5151],
#                 [34.4439, 32.9391],
#                 [66.2357, 31.6240],
#                 [69.9224, 28.6391],
#                 [74.6844, 27.9160],
#                 [78.4896, 28.7845],
#                 [81.7763, 30.7777],
#                 [78.5424, 32.3643],
#                 [74.9500, 33.0701],
#                 [70.4815, 32.7997],
#                 [40.5623, 69.4264],
#                 [46.7165, 66.6877],
#                 [53.9384, 65.6637],
#                 [57.0367, 65.7149],
#                 [60.1884, 65.5785],
#                 [67.0421, 66.4981],
#                 [72.8433, 68.9852],
#                 [68.8455, 73.6112],
#                 [63.5241, 77.1335],
#                 [57.1181, 78.4179],
#                 [50.4625, 77.4115],
#                 [44.8436, 74.0795],
#                 [41.7917, 69.6617],
#                 [49.2115, 68.7197],
#                 [57.0720, 68.7140],
#                 [64.5697, 68.5237],
#                 [71.6664, 69.2040],
#                 [64.8789, 72.3259],
#                 [57.0472, 73.6515],
#                 [48.9121, 72.6308],
#                 [38.7851, 30.8744],
#                 [74.4489, 30.4575]])
# np.save('data/wflw/init_landmark.npy', coord)

# 300w
# coord = np.array([[13.9787, 32.1156],
#         [14.2668, 43.3620],
#         [15.6831, 54.5938],
#         [18.1842, 65.4843],
#         [22.6926, 75.5516],
#         [29.5643, 84.3115],
#         [37.9050, 91.6050],
#         [47.1645, 97.3576],
#         [57.5145, 98.7971],
#         [67.6948, 96.8463],
#         [76.6377, 90.8428],
#         [84.7539, 83.4179],
#         [91.3325, 74.5627],
#         [95.6196, 64.3052],
#         [97.7914, 53.2779],
#         [98.8152, 42.1016],
#         [98.9689, 30.8809],
#         [21.3846, 23.6313],
#         [26.7282, 18.6939],
#         [34.2579, 17.2842],
#         [42.0711, 18.3436],
#         [49.1439, 21.3522],
#         [62.4795, 20.7545],
#         [69.9200, 17.6094],
#         [77.6691, 16.4948],
#         [85.2134, 17.8993],
#         [90.4471, 22.6043],
#         [56.1734, 29.8037],
#         [56.2737, 37.0641],
#         [56.3763, 44.2712],
#         [56.5322, 51.6990],
#         [47.6243, 56.7393],
#         [51.9854, 58.2864],
#         [56.5135, 59.5459],
#         [61.0422, 58.1508],
#         [65.0965, 56.6548],
#         [30.1086, 31.0749],
#         [34.7612, 28.3360],
#         [40.3245, 28.4449],
#         [45.1003, 31.9787],
#         [40.1559, 32.9803],
#         [34.5935, 33.0330],
#         [66.9809, 31.6408],
#         [71.9343, 27.8279],
#         [77.4316, 27.7717],
#         [81.9825, 30.3595],
#         [77.9073, 32.3138],
#         [72.4614, 32.5113],
#         [40.1534, 70.4509],
#         [46.3493, 67.8177],
#         [52.4519, 66.5265],
#         [56.4528, 67.5576],
#         [60.9990, 66.4459],
#         [67.2359, 67.7005],
#         [73.2458, 69.8927],
#         [67.5186, 75.8977],
#         [61.6008, 78.5929],
#         [56.7186, 79.1547],
#         [52.2918, 78.7609],
#         [46.2579, 76.2988],
#         [42.6929, 70.7221],
#         [52.4408, 70.1502],
#         [56.5262, 70.5348],
#         [61.1197, 70.0108],
#         [70.7209, 70.2769],
#         [61.2309, 73.1178],
#         [56.5580, 73.7348],
#         [52.3823, 73.3020]])

# coord_ref = np.load('data/init_landmark.npy')
# coord_ref -= 56
# coord_ref *= 1.25
# coord_ref += 56
# for i in range(68):
#     heatmap = generate_target(np.zeros([256, 256]), coord_ref[i] * 256 / 112, sigma=5)
#     imageio.imwrite('heatmap_img/{}.jpg'.format(i), (heatmap*255).astype(np.uint8))
# fig = plt.figure()
# fig.set_size_inches(1, 1, forward=False)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# plt.imshow(heatmap)
# plt.savefig('test.jpg')

# coord = scipy.io.loadmat('data/300w/upsample_131.mat')['upsample_131']
# coord = scipy.io.loadmat('data/300w/images/helen/Helen_meanShape_256_1_5x.mat')['Helen_meanShape_256_1_5x']
# coord = scipy.io.loadmat('data/300w/images/helen/300w_to_Helen_finalShape_fineTuned.mat')['data_300w']
# coord = scipy.io.loadmat('data/300w/Helen_to_300w_finalShape_fineTuned.mat')['data_Helen']
# coord = scipy.io.loadmat('data/300w/WFLW_to_300w_finalShape_fineTuned.mat')['data_WFLW']
# coord = scipy.io.loadmat('data/wflw/300w_to_WFLW_finalShape_fineTuned.mat')['data_300w']
# coord = scipy.io.loadmat('data/300w/WFLW_to_300w_finalShape_withoutEyeCenter.mat')['shape_final']
# coord *= (112 / 256)
# coord -= 56
# coord *= 1.25
# coord += 56
# coord = np.ma.array(coord, mask=False)
# coord.mask[68] = True
# coord.mask[77] = True
# x = coord[:, 0]
# y = coord[:, 1]
# plt.scatter(x, y)
# plt.savefig('300wtoWFLW_finetuned.jpg')


img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = np.load('data/init_landmark.npy')   
tpts -= 56
tpts *= 1.25
tpts += 56
tpts *= (256 / 112)
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/300w_68.jpg', img.astype(np.uint8))
print('meanface/300w_68.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
        27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]
tpts = tpts[index]
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/300w_46.jpg', img.astype(np.uint8))
print('meanface/300w_46.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = np.load('data/wflw/init_landmark.npy')
tpts *= (256 / 112)
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/wflw_98.jpg', img.astype(np.uint8))
print('meanface/wflw_98.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 17, 2)) + list(np.arange(18, 33, 2)) + [33, 35, 37, 38, 40] +\
        [42, 44, 46, 48, 50] + list(np.arange(51, 55)) + list(np.arange(55, 60, 2)) +\
        list(np.arange(60, 68, 2)) + list(np.arange(68, 76, 2)) + list(np.arange(76, 83, 2)) +\
        list(np.arange(83, 88, 2)) + list(np.arange(88, 93, 2)) + [93, 95]
tpts = tpts[index]
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/wflw_54.jpg', img.astype(np.uint8))
print('meanface/wflw_54.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
tpts = scipy.io.loadmat('data/300w/images/helen/Helen_meanShape_256_1_5x.mat')['Helen_meanShape_256_1_5x']
tpts = tpts.astype(np.uint8)
for k in range(tpts.shape[0]):
    cv2.circle(img, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_194.jpg', img.astype(np.uint8))
print('meanface/helen_194.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 21, 2)) + [21] + list(np.arange(24, 41, 2)) + list(np.arange(41, 58, 2)) +\
        list(np.arange(58, 72, 2)) + list(np.arange(72, 86, 2)) + list(np.arange(86, 100, 2)) +\
        list(np.arange(100, 114, 2)) + list(np.arange(114, 134, 2)) + list(np.arange(134, 154, 2)) +\
        list(np.arange(154, 174, 2)) + list(np.arange(174, 194, 2))
tpts1 = tpts[index]
tpts1 = tpts1.astype(np.uint8)
for k in range(tpts1.shape[0]):
    cv2.circle(img, (tpts1[k, 0], tpts1[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_98.jpg', img.astype(np.uint8))
print('meanface/helen_98.jpg')

img = (np.zeros((256, 256, 3))+255).astype(np.float32)
index = list(np.arange(0, 21, 3)) + [20, 21] + list(np.arange(25, 41, 3)) + [41, 44, 47, 49, 51, 54, 57] +\
        [58, 61, 64, 65, 68, 71] + [72, 75, 78, 79, 82, 85] + [86, 89, 92, 93, 96, 99] +\
        [100, 103, 106, 107, 110, 113] + [114, 116, 119, 122, 124, 126, 129, 132] +\
        [134, 136, 139, 142, 144, 146, 149, 152] + [154, 156, 159, 162, 164, 166, 169, 162] +\
        [174, 176, 179, 182, 184, 186, 189, 192]
tpts1 = tpts[index]
tpts1 = tpts1.astype(np.uint8)
for k in range(tpts1.shape[0]):
    cv2.circle(img, (tpts1[k, 0], tpts1[k, 1]), 3, [0, 255, 0], -1)
imageio.imwrite('meanface/helen_78.jpg', img.astype(np.uint8))
print('meanface/helen_78.jpg')
