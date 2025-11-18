import os
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import img2pdf
import fitz
from vllm import LLM, SamplingParams
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, MODEL_PATH, GPU_UTILIZATION, PARALLEL_SIZE
from utils import re_match, process_image_with_refs

import asyncio

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
generate_semaphore = asyncio.Semaphore(1) 

# 初始化模型
# 用 asyncio 控制一下, 调用 generate 方法的协程数量
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    trust_remote_code=True,
    max_model_len=8192,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=PARALLEL_SIZE,
    gpu_memory_utilization=GPU_UTILIZATION,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

def pdf_to_images_high_quality(pdf_path, dpi=144):
    pdf = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    for page in pdf:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    pdf.close()
    return images

def pil_to_pdf_img2pdf(pil_images, output_path):
    img_bytes = [io.BytesIO() for _ in pil_images]
    for buf, img in zip(img_bytes, pil_images):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=95)
    pdf_bytes = img2pdf.convert([b.getvalue() for b in img_bytes])
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

def process_single_image(image):
    return {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=CROP_MODE
            )
        },
    }

async def process_pdf_file(input_pdf, output_dir, r, task_id):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    images = pdf_to_images_high_quality(input_pdf) # pdf -> imgs
    

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(executor.map(process_single_image, images), total=len(images)))

    async with generate_semaphore:
        r.hset(f'task:{task_id}', 'status', 'processing')
        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    output_path = output_dir
    os.makedirs(output_path, exist_ok=True)

    name_mmd_det = input_pdf.split('/')[-1].replace('.pdf', '_det.mmd')
    name_mmd = input_pdf.split('/')[-1].replace('pdf', 'mmd')
    name_pdf_out = input_pdf.split('/')[-1].replace('.pdf', '_layouts.pdf')

    mmd_det_path = output_path + '/' + name_mmd_det
    mmd_path = output_path + '/' + name_mmd
    pdf_out_path = output_path + '/' + name_pdf_out
    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0
    for output, img in zip(outputs_list, images): # 每一个 image 出来的 output 对应上
        content = output.outputs[0].text

        if '<｜end▁of▁sentence｜>' in content: # repeat no eos
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if SKIP_REPEAT:
                continue

        page_num = f'\n<--- Page Split --->'
        contents_det += content + f'\n{page_num}\n'
        
        image_draw = img.copy()
        matches_ref, matches_images, mathes_other = re_match(content)
        # print(matches_ref)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx)

        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(mathes_other):
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'
        jdx += 1

    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    r.hset(f'task:{task_id}', 'status', 'fulfilled')

    return {"mmd": name_mmd_det, "pdf": name_pdf_out}
