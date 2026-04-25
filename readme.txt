我来帮你查看 EndoDAC 项目下各个脚本的作用。首先让我列出目录结构并查看主要脚本文件。

[13 tools called]

根据我对 EndoDAC 项目的分析，这是一个用于内窥镜手术场景的自监督深度估计框架。以下是各个脚本的作用说明：

## 主要训练和测试脚本

**1. `train_end_to_end.py`**
- **作用**：训练脚本入口
- **功能**：初始化训练器并启动端到端训练流程，设置随机种子确保可重复性

**2. `trainer_end_to_end.py`**
- **作用**：核心训练器类
- **功能**：包含完整的训练逻辑，包括模型初始化、数据加载、损失计算、优化器设置等

**3. `test_simple.py`**
- **作用**：简单的深度预测测试脚本
- **功能**：对单张图片或图片文件夹进行深度估计推理，支持 EndoDAC 和 AF-SfM 两种模型，输出深度视觉化结果和 .npy 文件

## 评估脚本

**4. `evaluate_depth.py`**
- **作用**：深度估计评估
- **功能**：在测试集上评估训练好的深度估计模型，计算各种深度估计指标（abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3）

**5. `evaluate_pose.py`**
- **作用**：位姿估计评估
- **功能**：评估相机位姿估计性能，计算轨迹误差（ATE）和旋转误差（RE），支持相机内参学习

**6. `mytest_da.py`**
- **作用**：Depth Anything 模型测试
- **功能**：使用预训练的 Depth Anything 模型进行深度估计评估，作为基线对比

## 数据处理脚本

**7. `export_gt_depth.py`**
- **作用**：导出真值深度数据
- **功能**：从 SCARED 数据集中提取地面真值深度图，保存为 .npz 格式供评估使用

**8. `export_gt_pose.py`**
- **作用**：导出真值位姿数据
- **功能**：从 SCARED 数据集的 JSON 文件中提取相机位姿真值，用于位姿评估

## 可视化脚本

**9. `visualize_pose.py`**
- **作用**：位姿轨迹可视化
- **功能**：绘制预测的相机轨迹与真值轨迹的 3D 对比图

**10. `visualize_reconstruction.py`**
- **作用**：3D 重建可视化
- **功能**：使用预测的深度图和相机内参重建 3D 点云，并保存为 .ply 格式

**11. `generate_3d_video.py`**
- **作用**：生成 3D 点云旋转视频
- **功能**：读取点云文件，通过旋转视角生成 MP4 视频用于展示

## 配置和工具

**12. `options.py`**
- **作用**：命令行参数配置
- **功能**：定义所有训练、测试、评估的超参数和选项

**13. `requirements.txt`**
- **作用**：Python 依赖列表
- **功能**：列出项目所需的所有 Python 包

**14. `README.md`**
- **作用**：项目说明文档
- **功能**：介绍项目背景、使用方法、结果对比等

## 使用流程总结

1. **训练**: `train_end_to_end.py` → `trainer_end_to_end.py`
2. **导出真值**: `export_gt_depth.py` + `export_gt_pose.py`
3. **评估**: `evaluate_depth.py` + `evaluate_pose.py`
4. **推理**: `test_simple.py`
5. **可视化**: `visualize_pose.py` + `visualize_reconstruction.py` + `generate_3d_video.py`

这是一个完整的内窥镜深度估计框架，采用基础模型适配（LoRA）技术，实现高效的自监督学习。

--load_weights_folder - 指定模型权重路径
--eval_split custom - 使用 custom split
--eval_mono - 单目评估模式
--model_type endodac - 模型类型


------------------------------------------------------------------------------运行步骤

第一步，首先在splits/endovis下生成train_files.txt、test_files.txt、val_files.txt文件。

第二步，在splits/endovis下用export_gt_depth.py生成gt_depths.npz需要图像的深度真值.tiff
python export_gt_depth.py --data_path /media/hoo/新加卷/yinghui/SCARED --split endovis --useage eval
注意：数据集的格式SCARED下分为train和test。如：/media/hoo/新加卷/yinghui/SCARED/train/dataset1/keyframe_1/data/left

第四步，开始训练。
 CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path /media/hoo/新加卷/yinghui/SCARED --log_dir './logs' --model_name EndoDAC --num_epochs 20 --batch_size 4  --dataset endovis --split endovis

--split X 参数可选，只要在在splits/X下生成train_files.txt、test_files.txt、val_files.txt文件

第五步，评估模型。
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path /media/hoo/新加卷/yinghui/SCARED --eval_split endovis --load_weights_folder ./logs/EndoDAC/models/weights_last --eval_mono --visualize_depth --visualize_depth会在你评估的模型里生成可视化。

第六步，用模型生成深度图。
 python test_simple.py --image_path /home/hoo/X/Endoscope-video/hamlyn_data/rectified01/image01 --model_path ./logs/EndoDAC/models/weights_last  --ext jpg --output_path /home/hoo/X/Endoscope-video/hamlyn_data/rectified01/dac
--save_npy 生成归一化视差npy文件。

--------------------------------------------------------------位姿
第一步，生成pose.npz。需要.json真值文件，需要在splits/endovis下生成test_files_sequence1.txt、test_files_sequence2.txt文件。生成的npz位于splits/endovis/curve。
CUDA_VISIBLE_DEVICES=0 python export_gt_pose.py --data_path /media/hoo/新加卷/yinghui/SCARED --split endovis --sequence sequence1
--sequence sequenceX 可选

第二步，预测pose。预测的pose是npz格式。如需四元数形式运行export_pred_poses_quat.py
CUDA_VISIBLE_DEVICES=0 python evaluate_pose.py --data_path /media/hoo/新加卷/yinghui/SCARED --eval_split endovis --load_weights_folder ./logs/EndoDAC/models/weights_last

命令端预测的K是归一化参数，如果要恢复到像素单位，可以乘回图像分辨率：
fx_pixels ≈ 0.9164 * width
fy_pixels ≈ 1.1513 * height
cx_pixels ≈ 0.4975 * width
cy_pixels ≈ 0.5008 * height

--------------------------------------------------------------重建
第一步，在splits/endovis下生成3d_reconstruction.txt，脚本会根据3d_reconstruction.txt找到需要重建的图像。
CUDA_VISIBLE_DEVICES=0 python visualize_reconstruction.py --data_path /media/hoo/新加卷/yinghui/SCARED --eval_split endovis --load_weights_folder ./logs/EndoDAC/models/weights_last --eval_mono --save_recon



插入RFA模块   改了endodac.py   评估的时候也要用对应的endodac.py
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py  --data_path /media/hoo/新加卷/yinghui/SCARED   --eval_split endovis_deblur --load_weights_folder ./logs/EndoDAC_deblur/models/weights_last --eval_mono --visualize_depth --visualize_depth


CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py  --data_path /media/hoo/新加卷/yinghui/SCARED  --log_dir './logs' --model_name EndoDAC_deblur_RFA_proj_before_20 --num_epochs 20  --batch_size 4  --dataset endovis --split endovis_deblur --use_wandb True --wandb_project EndoDAC_deblur_ablation --wandb_run_name EndoDAC_RFA_proj_before_20


WTconV替换conV 改了layer.py

CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py  --data_path /media/hoo/新加卷/yinghui/SCARED  --log_dir './logs' --model_name EndoDAC_WTConv --num_epochs 20  --batch_size 4  --dataset endovis --split endovis_blur --use_wandb True --wandb_project EndoDAC_deblur_ablation --wandb_run_name EndoDAC_WTConv_20


PCC
CUDA_VISIBLE_DEVICES=0 python train_end_to_end_PCC.py --data_path /media/hoo/新加卷/yinghui/SCARED --log_dir ./logs --model_name debug_PCC --num_epochs 1 --batch_size 4 --dataset endovis --split endovis_deblur --use_wandb true --wandb_project EndoDAC_deblur_ablation --wandb_run_name debug_PCC --use_pcc True --pcc_weight 0.005 --pcc_use_cycle True --pcc_cycle_weight 0.5 --pcc_window 5

CUDA_VISIBLE_DEVICES=0 python train_end_to_end_PCC.py --data_path /media/hoo/新加卷/yinghui/SCARED --log_dir ./logs --model_name EndoDAC_deblur_RFA_proj_after_20_PCC --num_epochs 20 --batch_size 4 --dataset endovis --split endovis_deblur --use_wandb True --wandb_project EndoDAC_deblur_ablation --wandb_run_name EndoDAC_RFA_proj_after_20_PCC --use_pcc True --pcc_weight 0.01 --pcc_use_cycle True --pcc_cycle_weight 0.5 --pcc_window 5



