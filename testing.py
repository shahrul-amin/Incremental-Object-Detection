#!/usr/bin/env python3

import json
import argparse
import os
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.utils.logger import setup_logger
from pycocotools.coco import COCO
import logging

def main():
    parser = argparse.ArgumentParser(description="Detectron2 style COCO evaluation")
    parser.add_argument('gt_json', help="Ground truth JSON file")
    parser.add_argument('pred_json', help="Predictions JSON file")
    
    args = parser.parse_args()
    
    # Setup logger to get the exact Detectron2 format
    setup_logger()
    logger = logging.getLogger("detectron2.evaluation.fast_eval_api")
    
    print(f"Ground truth: {args.gt_json}")
    print(f"Predictions: {args.pred_json}")
    
    # Load COCO ground truth
    coco_gt = COCO(args.gt_json)
    
    coco_dt = coco_gt.loadRes(args.pred_json)
    
    logger.info("Evaluate annotation type *bbox*")
    coco_eval = COCOeval_opt(coco_gt, coco_dt, 'bbox')
    
    import time
    start_time = time.time()
    coco_eval.evaluate()
    eval_time = time.time() - start_time
    logger.info(f"COCOeval_opt.evaluate() finished in {eval_time:.2f} seconds.")
    
    logger.info("Accumulating evaluation results...")
    start_time = time.time()
    coco_eval.accumulate()
    accum_time = time.time() - start_time
    logger.info(f"COCOeval_opt.accumulate() finished in {accum_time:.2f} seconds.")
    
    coco_eval.summarize()
    
    logger = logging.getLogger("detectron2.evaluation.coco_evaluation")
    logger.info("Evaluation results for bbox:")
    
    stats = coco_eval.stats
    
    print("|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |")
    print("|:------:|:------:|:------:|:------:|:------:|:------:|")
    print(f"| {stats[0]*100:.3f} | {stats[1]*100:.3f} | {stats[2]*100:.3f} | {stats[3]*100:.3f} | {stats[4]*100:.3f} | {stats[5]*100:.3f} |")
    
    logger.info("Per-category bbox AP:")
    print("| category   | AP     | category   | AP     | category   | AP     |")
    print("|:-----------|:-------|:-----------|:-------|:-----------|:-------|")
    
    cats = coco_gt.cats
    cat_ids = sorted(cats.keys())
    
    # Calculate per-category AP
    precisions = coco_eval.eval['precision']
    per_cat_ap = []
    
    for idx, cat_id in enumerate(cat_ids):
        ap = precisions[:, :, idx, 0, -1]
        ap = ap[ap > -1] 
        ap = ap.mean() if len(ap) > 0 else 0.0
        
        cat_name = cats[cat_id]['name']
        per_cat_ap.append((cat_name, ap))
    
    for i in range(0, len(per_cat_ap), 3):
        row_parts = []
        for j in range(3):
            if i + j < len(per_cat_ap):
                cat_name, ap = per_cat_ap[i + j]
                row_parts.extend([f"{cat_name:<10}", f"{ap*100:.3f}"])
            else:
                row_parts.extend([" " * 10, " " * 6])
        print(f"| {row_parts[0]} | {row_parts[1]} | {row_parts[2]} | {row_parts[3]} | {row_parts[4]} | {row_parts[5]} |")
    
    from collections import OrderedDict
    result = OrderedDict([
        ('bbox', {
            'AP': stats[0] * 100,
            'AP50': stats[1] * 100,
            'AP75': stats[2] * 100,
            'APs': stats[3] * 100,
            'APm': stats[4] * 100,
            'APl': stats[5] * 100,
        })
    ])
    
    for cat_name, ap in per_cat_ap:
        result['bbox'][f'AP-{cat_name}'] = ap * 100
    
    print(result)

if __name__ == "__main__":
    main()
