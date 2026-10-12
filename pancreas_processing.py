"""
预处理 NIH Pancreas-CT (TCIA) 数据集，生成训练/测试所需的数据。

输入（原始数据，未预处理）:
    data/Pancreas-CT/
        pancreas_ct/PANCREAS_XXXX/.../Pancreas-XXXXX/*.dcm      # CT 的 DICOM 序列
        TCIA_pancreas_labels-02-05-2017-1/TCIA_pancreas_labels-02-05-2017/labelXXXX.nii.gz  # 分割标签(0/1)

输出（训练脚本 ad_ema_mix9.py 直接读取）:
    data/Pancreas/
        Pancreas_h5/PANCREAS_XXXX_norm.h5    # 每个样本一个 h5, 含 'image'(float32) 与 'label'(uint8 0/1)
        train.list                           # 训练样本名(PANCREAS_XXXX)
        test.list                            # 测试样本名

处理流程（与 LA 的 la_heart_processing.py 对应, 参考 SSL4MIS 的 Pancreas 预处理）:
    1. SimpleITK 读 DICOM 序列 -> 3D 数组 (z,y,x); SimpleITK 读 label.nii.gz -> 同样 (z,y,x)。
       两者尺寸一致(逐切片一一对应), 因此直接按索引对齐, 无需重采样。
    2. CT 灰度截断到 [HU_MIN, HU_MAX] 并归一化到 [0,1]。
    3. 按 label 的非零包围盒裁剪 image/label, 四周留 MARGIN 个 voxel 的上下文。
    4. 每维补齐到至少 MIN_SIZE(96), 保证训练时 RandomCrop(96^3) 不会越界。
"""

import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
import h5py
from tqdm import tqdm

# Windows 默认 GBK 控制台无法输出中文/emoji, 统一转成 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==================== 可调参数 ====================
HU_MIN, HU_MAX = -125, 275        # CT 灰度截断窗口 (胰腺 CT 常用窗)
MARGIN = 25                       # 裁剪时围绕胰腺的边距 (voxel)
MIN_SIZE = 96                     # 每维最小尺寸 (与 patch_size 对齐)
NUM_TRAIN = 62                    # 训练样本数, 其余作为测试
# ================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'Pancreas-CT')
LABEL_DIR = os.path.join(RAW_DIR, 'TCIA_pancreas_labels-02-05-2017-1', 'TCIA_pancreas_labels-02-05-2017')
CT_DIR = os.path.join(RAW_DIR, 'pancreas_ct')
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'Pancreas')
H5_DIR = os.path.join(OUT_DIR, 'Pancreas_h5')


def read_ct_series(case_id):
    """读一个样本的 DICOM 序列, 返回 sitk.Image (保留空间信息)。"""
    case_dir = os.path.join(CT_DIR, case_id)
    # 递归找到最深的含 dcm 的目录
    dcm_dir = case_dir
    for root, dirs, files in os.walk(case_dir):
        if any(f.endswith('.dcm') for f in files):
            dcm_dir = root
            break
    # 用 SimpleITK 自动按 ImagePositionPatient 排序, 避免文件名排序错乱
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
    if not series_ids:
        raise FileNotFoundError(f'{case_id}: 未在 {dcm_dir} 找到 DICOM 序列')
    file_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image


def read_label(case_id):
    """读 labelXXXX.nii.gz, 返回 sitk.Image (保留空间信息)。"""
    num = case_id.split('_')[-1]
    path = os.path.join(LABEL_DIR, f'label{num}.nii.gz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{case_id}: 未找到标签 {path}')
    return sitk.ReadImage(path)


def align_label_to_ct(ct_image, label_image):
    """将 label 重采样到 CT 的空间坐标系(origin/spacing/direction), 保证严格对齐。
    label 用最近邻插值, 避免引入非 0/1 的值。"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(label_image)


def crop_with_margin(image, label, margin=MARGIN, min_size=MIN_SIZE):
    """按 label 非零区域裁剪 image/label, 并补齐到至少 min_size。"""
    nz = np.nonzero(label)
    z0, z1 = nz[0].min(), nz[0].max()
    y0, y1 = nz[1].min(), nz[1].max()
    x0, x1 = nz[2].min(), nz[2].max()

    Z, Y, X = label.shape
    z0 = max(z0 - margin, 0); z1 = min(z1 + margin + 1, Z)
    y0 = max(y0 - margin, 0); y1 = min(y1 + margin + 1, Y)
    x0 = max(x0 - margin, 0); x1 = min(x1 + margin + 1, X)

    image = image[z0:z1, y0:y1, x0:x1]
    label = label[z0:z1, y0:y1, x0:x1]

    # 补齐到 min_size (居中补 0), 避免训练 RandomCrop 越界
    pad = []
    for s in label.shape:
        d = min_size - s
        pad.append((max(d // 2, 0), max(d - d // 2, 0)) if d > 0 else (0, 0))
    image = np.pad(image, pad, mode='constant', constant_values=0)
    label = np.pad(label, pad, mode='constant', constant_values=0)
    return image, label


def preprocess():
    os.makedirs(H5_DIR, exist_ok=True)

    # 收集所有样本, 按编号排序 (保证切分稳定)
    cases = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CT_DIR, 'PANCREAS_*')))
    print(f'共发现 {len(cases)} 个样本')

    for case in tqdm(cases, desc='预处理'):
        ct_image = read_ct_series(case)
        label_image = read_label(case)

        # 将 label 重采样到 CT 空间, 保证 origin/spacing/direction 严格对齐
        label_image = align_label_to_ct(ct_image, label_image)

        # 转 numpy 数组 (z,y,x)
        image = sitk.GetArrayFromImage(ct_image).astype(np.float32)
        label = sitk.GetArrayFromImage(label_image)
        label = (label > 0).astype(np.uint8)
        assert image.shape == label.shape, f'{case}: CT {image.shape} 与 label {label.shape} 尺寸不一致'

        # 灰度截断 + 归一化到 [0,1]
        image = np.clip(image, HU_MIN, HU_MAX)
        image = (image - HU_MIN) / (HU_MAX - HU_MIN)
        image = image.astype(np.float32)

        # 裁剪 + 补齐
        image, label = crop_with_margin(image, label)

        out = os.path.join(H5_DIR, f'{case}_norm.h5')
        with h5py.File(out, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
            f.create_dataset('label', data=label, compression='gzip')

    # 生成 train/test 切分 (前 NUM_TRAIN 训练, 其余测试)
    train = cases[:NUM_TRAIN]
    test = cases[NUM_TRAIN:]
    with open(os.path.join(OUT_DIR, 'train.list'), 'w') as f:
        f.write('\n'.join(train) + '\n')
    with open(os.path.join(OUT_DIR, 'test.list'), 'w') as f:
        f.write('\n'.join(test) + '\n')

    print(f'\n✅ 预处理完成: {len(cases)} 个样本')
    print(f'   train: {len(train)} ({train[0]} ~ {train[-1]})')
    print(f'   test : {len(test)} ({test[0]} ~ {test[-1]})')
    print(f'   输出目录: {H5_DIR}')


if __name__ == '__main__':
    preprocess()