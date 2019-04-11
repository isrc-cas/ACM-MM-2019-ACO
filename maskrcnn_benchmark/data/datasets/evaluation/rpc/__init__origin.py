import json
import logging
import os
from datetime import datetime

import boxx
import rpctool
from tqdm import tqdm


def rpc_evaluation(dataset, predictions, output_folder, box_only, **_):
    print('no density map eval')
    pred_boxlists = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        bboxes = prediction.bbox.numpy()
        labels = prediction.get_field("labels").numpy()
        scores = prediction.get_field("scores").numpy()

        for i in range(len(prediction)):
            score = scores[i]
            box = bboxes[i]
            label = labels[i]

            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]

            pred_boxlists.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
            })

    if len(pred_boxlists) == 0:
        print('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'w') as fid:
            fid.write('Nothing detected.')
        return 'Nothing detected.'

    save_path = os.path.join(output_folder, 'bbox_results.json')
    with open(save_path, 'w') as fid:
        json.dump(pred_boxlists, fid)
    res_js = boxx.loadjson(save_path)
    ann_js = boxx.loadjson(dataset.annopath)
    result = rpctool.evaluate(res_js, ann_js)
    print(result)
    with open(os.path.join(output_folder, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'w') as fid:
        fid.write(str(result))
    return result


def __rpc_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    pred_boxlists = []
    annotations = []
    correct = 0
    mae = 0
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        bboxes = prediction.bbox.numpy()
        labels = prediction.get_field("labels").numpy()
        scores = prediction.get_field("scores").numpy()
        ann = dataset.get_annotation(img_info['id'])
        density_map = prediction.get_field('density_map').numpy()
        assert density_map.shape[0] == 1
        density = density_map.sum()
        if round(density) == len(ann):
            correct += 1
        mae += abs(density - len(ann))

        # -----------------------------------------------#
        # -----------------Pseudo Label------------------#
        # -----------------------------------------------#
        image_result = {
            'bbox': [],
            'width': image_width,
            'height': image_height,
            'id': img_info['id'],
            'file_name': img_info['file_name'],
        }

        for i in range(len(prediction)):
            score = scores[i]
            box = bboxes[i]
            label = labels[i]
            if score > 0.95:
                x, y, width, height = float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])
                image_result['bbox'].append(
                    (int(label), x, y, width, height)
                )
        if len(image_result['bbox']) >= 3 and len(image_result['bbox']) == round(density):
            annotations.append(image_result)
        # -----------------------------------------------#
        # -----------------------------------------------#
        # -----------------------------------------------#

        for i in range(len(prediction)):
            score = scores[i]
            box = bboxes[i]
            label = labels[i]

            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]

            pred_boxlists.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
            })

    print('Ratio: {}'.format(correct / len(predictions)))
    print('MAE: {:.3f} '.format(mae / len(predictions)))
    if len(pred_boxlists) == 0:
        print('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'w') as fid:
            fid.write('Nothing detected.')
        return 'Nothing detected.'

    print('Pseudo-Labeling: {}'.format(len(annotations)))
    with open(os.path.join(output_folder, 'pseudo_labeling.json'), 'w') as fid:
        json.dump(annotations, fid)
    save_path = os.path.join(output_folder, 'bbox_results.json')
    with open(save_path, 'w') as fid:
        json.dump(pred_boxlists, fid)
    res_js = boxx.loadjson(save_path)
    ann_js = boxx.loadjson(dataset.annopath)
    result = rpctool.evaluate(res_js, ann_js)
    print(result)
    with open(os.path.join(output_folder, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), 'w') as fid:
        fid.write(str(result) + '\n' + 'Ratio: {}, '.format(correct / len(predictions)) + 'MAE: {:.3f} '.format(mae / len(predictions)))
    return result
