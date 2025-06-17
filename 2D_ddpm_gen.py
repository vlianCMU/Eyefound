import os
import time
import math
from PIL import Image
import torch
from transformers import BertTokenizer, BertModel
from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers import UNet2DModel, UNet2DConditionModel
from my_pipeline_ddpm import DDPMPipeline

@dataclass
class GenerationConfig:
    image_size = 256
    eval_batch_size = 4
    mixed_precision = "fp16"
    output_dir = "./全新结果"  # 生成图像的保存目录
    pretrain_path = "./全新模型/unet/"  # 您训练好的模型路径
    bert_path = "./bertmodel"  # BERT模型路径
    seed = 42

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def generate_seed():
    seed = (
        time.time_ns()
        ^ (os.getpid() << 16)
    )
    return seed

def generate_images(config, condition_text, output_name, pipeline, tokenizer, bert_model, device):
    """生成特定条件文本的图像"""
    
    # BERT编码文本条件
    prompt_embeding = []
    print(f"生成条件: {condition_text}")
    inputs = tokenizer(condition_text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k:inputs[k].to(device) for k in inputs}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # 获取[CLS]标记的嵌入作为句子表示
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    prompt_embeding.append(cls_embedding.view(1, 1, -1))
    
    # 复制为batch_size大小
    prompt_embeding = prompt_embeding * config.eval_batch_size
    prompt_embeding = torch.cat(prompt_embeding, dim=0)
    
    # 生成图像
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(generate_seed()),
        condition=prompt_embeding,
    ).images
    
    # 创建网格并保存
    num_grid = int(math.sqrt(config.eval_batch_size))
    image_grid = make_grid(images, rows=num_grid, cols=num_grid)
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    grid_path = os.path.join(config.output_dir, f"{output_name}_grid.png")
    image_grid.save(grid_path)
    
    # 保存单独的图像
    for i, image in enumerate(images):
        img_path = os.path.join(config.output_dir, f"{output_name}_{i:03d}.png")
        image.save(img_path)
    
    print(f"图像已保存到: {config.output_dir}/{output_name}_*")
    return images

def gen_loop(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化BERT模型和分词器
    print(f"加载BERT模型: {config.bert_path}")
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = BertModel.from_pretrained(config.bert_path)
    bert_model = bert_model.to(device)
    
    # 初始化加速器
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    # 初始化UNet模型并加载预训练权重
    print(f"加载预训练模型: {config.pretrain_path}")
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512, 1024),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D",  
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=32,
        encoder_hid_dim_type="text_proj",
        encoder_hid_dim=768,
    )
    
    model = model.from_pretrained(config.pretrain_path)
    model = model.to(device)
    
    # 设置扩散调度器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建pipeline
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    
    # 定义要生成的眼底图像类型
    eye_conditions = [
        # ("normal", "a normal CFP image"),
        ("diabetic_retinopathy", "a colored fundus image with diabetic retinopathy"),
        ("glaucoma", "a colored fundus image with glaucoma"),
        ("cataract", "a colored fundus image with cataract"),
        ("age-related_macular_degeneration", "a colored fundus image with age-related macular degeneration"),
        

        # ("others", "a CFP image with no or other diseases")
        # ("others", "a CFP image that has no description or with no matching target diseases")

    ]
    
    # 为每种类型生成图像
    for condition_name, condition_text in eye_conditions:
        for i in range(1):  # 为每种类型生成5组图像，每组4张（根据eval_batch_size）
            unique_name = f"{condition_name}_set{i}"
            generate_images(config, condition_text, unique_name, pipeline, tokenizer, bert_model, device)
            
    print("所有图像生成完成!")

if __name__ == "__main__":
    config = GenerationConfig()
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 执行生成循环
    gen_loop(config)