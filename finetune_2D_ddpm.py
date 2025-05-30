import os
import pdb
import math
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from accelerate import Accelerator

from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler

from my_pipeline_ddpm import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.util import get_dl  # 使用原始的util.py
import safetensors
from transformers import BertTokenizer, BertModel

@dataclass
class FineTuneConfig:
    image_size = 256  
    train_batch_size = 24  # 可以根据显存调整
    eval_batch_size = 16
    num_epochs = 5  # finetune通常不需要很多epochs
    gradient_accumulation_steps = 1
    learning_rate = 5e-5  # finetune时学习率要比预训练小
    lr_warmup_steps = 500
    eval_freq_step = 1000
    mixed_precision = "fp16"
    output_dir = "finetune_cfp_model_epoch5_batch24"
    num_workers = 16
    overwrite_output_dir = True
    seed = 421
    bert_path = "./bertmodel"
    data_root = "../cfp_dataset_changgeng"  # 你的新数据路径
    
    # finetune特有配置
    pretrained_model_path = "./save_dir_new"  # 你的预训练模型路径
    save_steps = 1000
    validation_steps = 1000

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline, prompt_embeding):
    # 和原来一样的评估函数
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        condition=prompt_embeding,
    ).images

    num_grid = int(math.sqrt(config.eval_batch_size))
    image_grid = make_grid(images, rows=num_grid, cols=num_grid)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/finetune_{epoch:04d}.png")

def train_loop(config, accelerator, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    bert_path = config.bert_path
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("finetune_fundus_ddpm")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler, bert_model = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, bert_model
    )
    
    global_step = 0
    print("开始finetune训练...")
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # 和原来训练脚本完全一样的处理
            prompt_embeding = []
            for text in batch[1]:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                inputs = {k:inputs[k].to(batch[0].device) for k in inputs}
                with torch.no_grad():
                    outputs = bert_model(**inputs)
            
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                prompt_embeding.append(cls_embedding.view(1, 1, -1))
            prompt_embeding = torch.cat(prompt_embeding, dim=0)

            clean_images = batch[0][:,:3,:,:]

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=prompt_embeding, return_dict=False)[0]
                
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # 定期评估和保存
            if accelerator.is_main_process and global_step % config.eval_freq_step == 0:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                evaluate(config, epoch*len(train_dataloader)+step, pipeline, prompt_embeding)
                
                # 保存checkpoint
                save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                pipeline.save_pretrained(save_path)
                print(f"模型已保存到: {save_path}")

        # 每个epoch结束保存
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline.save_pretrained(config.output_dir)
            print(f"Epoch {epoch} 完成，模型已保存")

if __name__ == "__main__":
    config = FineTuneConfig()

    # 检查必要的路径
    assert os.path.exists(config.data_root), f"数据路径不存在: {config.data_root}"
    assert os.path.exists(config.pretrained_model_path), f"预训练模型路径不存在: {config.pretrained_model_path}"
    assert os.path.exists(config.bert_path), f"BERT模型路径不存在: {config.bert_path}"

    # 使用原来的数据加载器
    train_dataloader, train_dataset = get_dl(config=config)
    
    print(f"训练集样本数: {len(train_dataset)}")

    # 加载预训练的UNet模型
    print("加载预训练模型...")
    model = UNet2DConditionModel.from_pretrained(os.path.join(config.pretrained_model_path, "unet"))
    
    print(f"模型总参数数: {sum(p.numel() for p in model.parameters()):,}")

    # 测试数据加载
    sample_image, sample_text = train_dataset[0]
    print(f"样本图像形状: {sample_image.unsqueeze(0).shape}")
    print(f"样本文本: {sample_text}")

    # 初始化训练组件
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        device_placement=True
    )
    
    # 开始训练
    train_loop(config, accelerator, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)