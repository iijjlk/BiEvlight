import cv2
import random
import numpy as np
import torch

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_event_random_crop(img_gts, img_lqs, events_gts, events_lqs, lq_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(events_gts, list):
        events_gts = [events_gts]
    if not isinstance(events_lqs, list):
        events_lqs = [events_lqs]



    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]

    #crop events
    events_lqs_cropped = []
    for events in events_lqs:
        mask_x = torch.where((events[:, 2] < left + lq_patch_size) & (events[:, 2] >= left))
        event_x = torch.index_select(events, 0, mask_x[0])
        mask_y = torch.where((event_x[:, 1] < top + lq_patch_size) & (event_x[:, 1] >= top))
        event_y = torch.index_select(event_x, 0, mask_y[0])
        event = event_y.clone()
        event[:, 2] = event_y[:, 2] - left
        event[:, 1] = event_y[:, 1] - top
        events_lqs_cropped.append(event)

    events_gts_cropped = []
    for events in events_gts:
        mask_x = torch.where((events[:, 2] < left_gt + gt_patch_size) & (events[:, 2] >= left_gt))
        event_x = torch.index_select(events, 0, mask_x[0])
        mask_y = torch.where((event_x[:, 1] < top_gt + gt_patch_size) & (event_x[:, 1] >= top_gt))
        event_y = torch.index_select(event_x, 0, mask_y[0])
        event = event_y.clone()
        event[:, 2] = event_y[:, 2] - left_gt
        event[:, 1] = event_y[:, 1] - top_gt
        events_gts_cropped.append(event)



    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(events_gts_cropped) == 1:
        events_gts_cropped = events_gts_cropped[0]
    if len(events_lqs_cropped) == 1:
        events_lqs_cropped = events_lqs_cropped[0]
    return img_gts, img_lqs, events_gts_cropped, events_lqs_cropped


def paired_random_crop(img_gts, img_lqs, lq_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]


    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]



    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs



def paired_random_crop_with_voxels(img_gts, img_lqs, event_gts_voxel, event_lqs_voxel,cnt_lqs,cnt_gts, lq_patch_size, scale, gt_path):
    """Random crop that handles both images and voxelized events."""
    h_lq, w_lq, _ = img_lqs.shape
    h_gt, w_gt, _ = img_gts.shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq}).')

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = img_lqs[top:top + lq_patch_size, left:left + lq_patch_size, ...]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = img_gts[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]

    # Crop voxelized events accordingly
    event_lqs_voxel = event_lqs_voxel[:, top:top + lq_patch_size, left:left + lq_patch_size]
    event_gts_voxel = event_gts_voxel[:, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]

    cnt_lqs = cnt_lqs[:, top:top + lq_patch_size, left:left + lq_patch_size]
    cnt_gts = cnt_gts[:, top_gt:top_gt + gt_patch_size, left_gt:left + gt_patch_size]
    return img_gts, img_lqs, event_gts_voxel, event_lqs_voxel,cnt_lqs,cnt_gts

def paired_random_crop_DP(img_lqLs, img_lqRs, img_gts, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqLs, list):
        img_lqLs = [img_lqLs]
    if not isinstance(img_lqRs, list):
        img_lqRs = [img_lqRs]

    h_lq, w_lq, _ = img_lqLs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqLs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqLs
    ]

    img_lqRs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqRs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqLs) == 1:
        img_lqLs = img_lqLs[0]
    if len(img_lqRs) == 1:
        img_lqRs = img_lqRs[0]
    return img_lqLs, img_lqRs, img_gts


def paired_random_crop_with_eventflow( img_lqs,img_gts, event, lq_patch_size, scale, gt_path):
    """Random crop that handles both images and voxelized events."""
    h_lq, w_lq, _ = img_lqs.shape
    h_gt, w_gt, _ = img_gts.shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq}).')

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    min_x = random.randint(0, h_lq - gt_patch_size)
    min_y = random.randint(0, w_lq - gt_patch_size)

    max_y = min_y +gt_patch_size
    max_x = min_x +gt_patch_size

    # crop lq patch
    img_lqs = img_lqs[min_x:max_x, min_y:max_y, ...]
    img_gts = img_gts[min_x:max_x, min_y:max_y, ...]

    # crop voxel

    #event_denoising = event_denoising[:, min_x:max_x, min_y:max_y]


    # crop corresponding gt patch

    # Crop  events flows

    mask_x = torch.where((event[:, 2] < max_x) & (event[:, 2] >= min_x))
    event_x = torch.index_select(event, 0, mask_x[0])
    mask_y = torch.where((event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y))
    event_y = torch.index_select(event_x, 0, mask_y[0])
    event = event_y.clone()
    event[:, 2] = event_y[:, 2] - min_x
    event[:, 1] = event_y[:, 1] - min_y

    return img_lqs,img_gts, event



def paired_random_crop_esl( img_lqs,img_gts, event, lq_patch_size, scale, gt_path):
    """Random crop that handles both images and voxelized events."""
    h_lq, w_lq, _ = img_lqs.shape
    h_gt, w_gt, _ = img_gts.shape
    gt_patch_size = int(lq_patch_size * scale)

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        print(gt_path)
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq}).')

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    min_x = random.randint(0, h_lq - gt_patch_size)
    min_y = random.randint(0, w_lq - gt_patch_size)

    max_y = min_y +gt_patch_size
    max_x = min_x +gt_patch_size

    # crop lq patch
    img_lqs = img_lqs[min_x:max_x, min_y:max_y, ...]
    img_gts = img_gts[min_x:max_x, min_y:max_y, ...]

    # crop voxel
    # crop corresponding gt patch

    # Crop  events flows

    mask_x = torch.where((event[:, 2] < max_x) & (event[:, 2] >= min_x))
    event_x = torch.index_select(event, 0, mask_x[0])
    mask_y = torch.where((event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y))
    event_y = torch.index_select(event_x, 0, mask_y[0])
    event = event_y.clone()
    event[:, 2] = event_y[:, 2] - min_x
    event[:, 1] = event_y[:, 1] - min_y




    return img_lqs,img_gts, event


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


#裁剪flow_event


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def rotate_voxels(voxel_grid, angle, center):
    """Rotate the 3D voxel grid by a given angle around a center."""
    # angle is the rotation angle in degrees
    # center is the center of rotation in voxel coordinates

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Create a new voxel grid to store rotated results
    rotated_voxel_grid = np.zeros_like(voxel_grid)

    for x in range(voxel_grid.shape[1]):
        for y in range(voxel_grid.shape[2]):
            for t in range(voxel_grid.shape[0]):
                # Get the voxel coordinates (x, y, t)
                new_x = int(cos_angle * (x - center[0]) - sin_angle * (y - center[1]) + center[0])
                new_y = int(sin_angle * (x - center[0]) + cos_angle * (y - center[1]) + center[1])

                # Make sure the new coordinates are within bounds
                if 0 <= new_x < voxel_grid.shape[1] and 0 <= new_y < voxel_grid.shape[2]:
                    rotated_voxel_grid[t, new_x, new_y] = voxel_grid[t, x, y]

    return rotated_voxel_grid










def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out








def rotate_image(image, angle, center=None):
    """Rotate a 2D image by a given angle around a center."""
    # Ensure image is a 2D numpy array (i.e., the image is 2D)
    if len(image.shape) == 3:
        image = image[0]  # Assuming single-channel images for simplicity

    # Perform rotation (if center is None, rotate around the image center)
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    # Perform the rotation (here np.rot90 rotates by 90 degrees)
    rotated_image = np.rot90(image)  # You can replace this with other rotation logic if needed

    return rotated_image




# def random_augmentation_with_events(*args):
#     out = []
#     flag_aug = random.randint(0, 7)
#     for data in args:
#         if isinstance(data, tuple) and len(data) == 4:
#             # Assuming data is a tuple of (image_gt, image_lq, event_gt, event_lq)
#             img_gt, img_lq, event_gt, event_lq = data
#             out_img_gt = data_augmentation(img_gt, flag_aug).copy()
#             out_img_lq = data_augmentation(img_lq, flag_aug).copy()
#
#             # Apply voxel rotation to event streams
#             rotated_event_gt = rotate_voxels(event_gt, flag_aug * 45, center=(img_gt.shape[1] // 2, img_gt.shape[0] // 2))
#             rotated_event_lq = rotate_voxels(event_lq, flag_aug * 45, center=(img_lq.shape[1] // 2, img_lq.shape[0] // 2))
#
#             out.append((out_img_gt, out_img_lq, rotated_event_gt, rotated_event_lq))
#         else:
#             out.append(data_augmentation(data, flag_aug).copy())
#     return out

def random_augmentation_with_voxel(img_lq, img_gt,voxel_lq):
    """
    同时对 RGB 图像和 voxel 数据做一致的随机旋转
    """
    flag_aug = random.randint(0, 7)

    # RGB 图像旋转（用原来的 data_augmentation）
    img_gt_rot = data_augmentation(img_gt, flag_aug).copy()
    img_lq_rot = data_augmentation(img_lq, flag_aug).copy()

    # voxel 旋转（torch版本）
    voxel_lq_rot = voxel_augmentation_torch(voxel_lq, flag_aug)
    return img_lq_rot,img_gt_rot,voxel_lq_rot




def voxel_augmentation_torch(voxel: torch.Tensor, mode: int) -> torch.Tensor:
    """
    对 voxel 数据进行空间旋转/翻转，与 RGB 图像保持一致
    voxel: torch.Tensor [C, H, W]
    mode: 与 data_augmentation 相同 (0~7)
    """
    if voxel.ndim != 3:
        raise ValueError("voxel 必须是3维 [C, H, W]")

    if mode == 0:  # 原图
        out = voxel
    elif mode == 1:  # flip up and down
        out = torch.flip(voxel, dims=[1])
    elif mode == 2:  # rotate counterwise 90 degree
        out = voxel.permute(0, 2, 1).flip(dims=[1])  # 改成翻 dim=1
    elif mode == 3:  # rotate 90 degree and flip up and down
        out = voxel.permute(0, 2, 1).flip(dims=[1]).flip(dims=[1])
        # 等价于 permute + flip dim=1 + flip dim=1（这里第二次 flip 可以换成 flip dim=2，看具体想法）
    elif mode == 4:  # rotate 180 degree
        out = torch.flip(voxel, dims=[1, 2])
    elif mode == 5:  # rotate 180 degree and flip up and down
        out = torch.flip(voxel, dims=[2])
    elif mode == 6:  # rotate 270 degree
        # 270 逆时针 = 90 顺时针，等价于 permute + flip dim=2
        # 但我们要模拟 np.rot90(k=3)（逆时针270），所以 permute + flip dim=2 → 这里保持和numpy一致
        out = voxel.permute(0, 2, 1).flip(dims=[2])
    elif mode == 7:  # rotate 270 degree and flip up and down
        out = voxel.permute(0, 2, 1).flip(dims=[2]).flip(dims=[1])
    else:
        raise Exception("Invalid mode for voxel augmentation")

    return out



def rotate_events(events, angle_deg, sensor_size):
    angle = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
    xs, ys, ts, ps = events[0], events[1], events[2], events[3]
    cx, cy = sensor_size[1] / 2.0, sensor_size[0] / 2.0

    x_shifted = xs - cx
    y_shifted = ys - cy
    xs_new =  x_shifted * torch.cos(angle) - y_shifted * torch.sin(angle) + cx
    ys_new =  x_shifted * torch.sin(angle) + y_shifted * torch.cos(angle) + cy

    rotated_events = torch.stack([xs_new, ys_new, ts, ps], dim=0)
    return rotated_events



def random_augmentation_with_Flow(img_gt, img_lq, img_grad, event,event_gt, sensor_size):
    """
    同时对 RGB 图像和 voxel 数据 (以及事件流) 做一致的随机旋转/翻转
    """
    flag_aug = random.randint(0, 7)

    # 图像部分增强
    img_gt_aug    = data_augmentation(img_gt, flag_aug)
    img_lq_aug    = data_augmentation(img_lq, flag_aug)
    img_grad_aug  = data_augmentation(img_grad, flag_aug)

    # 事件流部分增强
    # 对应 mode 转换为角度（0, 90, 180, 270 度）
    mode_to_angle = {0: 0, 1: 0, 2: 90, 3: 90, 4: 180, 5: 180, 6: 270, 7: 270}
    angle = mode_to_angle[flag_aug]

    events_rot = rotate_events(event, angle_deg=angle, sensor_size=sensor_size)
    event_gt_rot = rotate_events(event_gt, angle_deg=angle, sensor_size=sensor_size)

    # flip ud模式需要上下翻转 y 坐标
    if flag_aug in [1, 3, 5, 7]:
        ys = events_rot[1]
        H = sensor_size[0]
        ys = H - 1 - ys
        events_rot[1] = ys

    return img_gt_aug, img_lq_aug, img_grad_aug, events_rot, event_gt_rot