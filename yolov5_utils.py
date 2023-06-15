import time
import torch
import torchvision
import numpy as np
import cv2
from typing import Iterable, List
import os

from ppq.core import convert_any_to_torch_tensor, ppq_warning

def load_calibration_dataset_yolov5(
    directory: str, input_shape: List[int],
    batchsize: int, input_format: str = 'chw',single = False) -> Iterable:
    """使用这个函数来加载校准数据集，校准数据集将被用来量化你的模型。这个函数只被用来加载图像数据集。
    你需要给出校准数据集位置，我们建议你将每张图片都单独保存到文件中，这个函数会自己完成后续的打包处理工作。
    校准数据集不应过大，这个函数会将所有数据加载到内存中，同时过大的校准数据集也不利于后续的量化处理操作。

    我们推荐你使用 512 ~ 4096 张图片进行校准，batchsize 设置为 16 ~ 64。
    我们支持读入 .npy 格式的数据，以及 .bin 或 .raw 的二进制数据，如果你选择以二进制格式输入数据，则必须指定样本尺寸
    如果你的样本尺寸不一致（即动态尺寸输入），则你必须使用 .npy 格式保存你的数据。

    如果这个函数无法满足你的需求，例如你的模型存在多个输入，则你可以自行构建数据集
    ppq 支持使用任何可遍历对象充当数据集，包括 torch.Dataset, list 等。

    Args:
        directory (str): 加载数据集的目录，目录不应包含子文件夹，所有目录中的文件将被视为数据。
        input_shape (List[int]): 图像尺寸，对于二进制输入文件而言，你必须指定图像尺寸，对于 npy文件 此项不起作用
        batchsize (int): batchsize 大小，这个函数会自动进行打包操作，但是如果你的数据本身已经有了预设的batchsize，
            则该函数不会覆盖原有batchsize
        input_format (str, optional): chw 或 hwc，指定输入图像数据排布。即使你的图像具有batch维度，它仍然将正常工作。

    Raises:
        FileNotFoundError: _description_
        ValueError: _description_

    Returns:
        Iterable: _description_
    """

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise FileNotFoundError(f'无法从指定位置加载数据集 {directory}. '
                                 '目录不存在或无法访问，检查你的输入路径')
    if input_format not in {'chw', 'hwc'}:
        raise ValueError(f'无法理解的数据格式，对于图片数据，数据格式只能是 chw 或 hwc，而你输入了 {input_format}')

    num_of_file, samples, sizes = 0, [], set()
    for file in os.listdir(os.path.join(directory, 'data')):
        sample = None
        if file.endswith('.npy'):
            sample = np.load(os.path.join(directory, 'data', file))
            num_of_file += 1
        elif file.endswith('.bin') or file.endswith('.raw'):
            sample = np.fromfile(os.path.join(directory, 'data', file), dtype=np.float32)
            assert isinstance(sample, np.ndarray), f'数据应当是 numpy.ndarray，然而你输入了 {type(sample)}'
            sample = sample.reshape(input_shape)
            num_of_file += 1
        elif file.endswith('.png') or file.endswith('.jpg'):
            if single == True:
                file = '3331fe62-color_421.png'
            im = cv2.imread(os.path.join(directory, 'data', file))
            h0, w0 = im.shape[:2]
            h, w =im.shape[:2]
            shape = im.shape[:2]
            new_shape = (input_shape[-2], input_shape[-1])
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            pad = (dw, dh)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            sample = im
            num_of_file += 1
        else:
            ppq_warning(f'文件格式不可读: {os.path.join(directory, "data", file)}, 该文件已经被忽略.')

        sample = convert_any_to_torch_tensor(sample)
        sample = sample.float()

        if sample.ndim == 3: sample = sample.unsqueeze(0)
        if input_format == 'hwc': sample = sample.permute([0, 3, 1, 2])

        if sample.shape[1] not in {1, 3, 4}:
            ppq_warning(f'Quantized data is not a regular channel, which channel is {sample.shape[1]}.')

        assert sample.shape[0] == 1 or batchsize == 1, (
            f'你的输入图像似乎已经有了预设好的 batchsize, 因此我们不会再尝试对你的输入进行打包。')

        sizes.add((sample.shape[-2], sample.shape[-1]))
        samples.append(sample)

    if len(sizes) != 1:
        ppq_warning('你的输入图像似乎包含动态的尺寸，因此 CALIBRATION BATCHSIZE 被强制设置为 1')
        batchsize = 1

    # create batches
    batches, batch = [], []
    if batchsize != 1:
        for sample in samples:
            if len(batch) < batchsize:
                batch.append(sample)
            else:
                batches.append(torch.cat(batch, dim=0))
                batch = [sample]
        if len(batch) != 0:
            batches.append(torch.cat(batch, dim=0))
    else:
        batches = samples

    print(f'{num_of_file} File(s) Loaded.')
    for idx, tensor in enumerate(samples[: 5]):
        print(f'Loaded sample {idx}, shape: {tensor.shape}')
    assert len(batches) > 0, '你送入了空的数据集'

    print(f'Batch Shape: {batches[0].shape}')
    return batches


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output