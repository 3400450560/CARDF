# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        
        # 执行非极大值抑制 (NMS)
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # 确保 orig_imgs 是列表，如果不是，则转换为列表
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # 计算 batch size (bs)
        f = lambda x: 4 if 4 <= x <= 16 else 4
        bs = f(orig_imgs[0].shape[-1])
        
        results, p = [], []

        # 遍历每一张图片和预测结果
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            
            # 调整预测框的坐标（根据图片大小缩放）
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img[...,-3:].shape)
            
            # 创建 Results 对象并添加到结果列表
            results.append(Results(orig_img[...,-3:], path=img_path, names=self.model.names, boxes=pred))

            # 处理 img_path 并确保安全性
            if 'ges' in img_path:
                batch = img_path.split('ges')
                batch = str(batch[0] + 'ge' + batch[1])  # 合并路径
            else:
                # 如果没有找到 'ges'，就直接使用原始路径
                batch = img_path
            
            # 根据图片的宽度判断是否满足条件，选择保存结果
            if orig_img.shape[-1] >= bs:
                p.append(Results(orig_img[...,:3], path=batch, names=self.model.names, boxes=pred))

        # 返回最终的结果
        return results, p if orig_img.shape[-1] >= bs else results
