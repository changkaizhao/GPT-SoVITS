import platform
import signal
import subprocess
import sys
import os
from multiprocessing import cpu_count

import traceback
from typing import List, Dict, Optional
from fastapi import APIRouter, Form, Request, UploadFile, File, HTTPException
import psutil
from pydantic import BaseModel
import shutil
import json
from datetime import datetime
import logging

import torch
import yaml
from config import (
    GPU_INDEX,
    GPU_INFOS,
    IS_GPU,
    infer_device,
    is_half,
    is_share,
    memset,
)
from GPT_SoVITS.inference import inference

from tools.i18n.i18n import I18nAuto, scan_language_list

i18n = I18nAuto()


os.environ["version"] = version = "v2Pro"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
torch.manual_seed(233333)

n_cpu = cpu_count()

set_gpu_numbers = GPU_INDEX
gpu_infos = GPU_INFOS
mem = memset
is_gpu_ok = IS_GPU

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
exp_root = "logs"
python_exec = "/home/roby/miniconda3/envs/GPTSoVits/bin/python"
tmp = ""

v3v4set = {"v3", "v4"}


def get_latest_gpt_weights(dir: str, id: str):
    """
    we have a folder path is dir
    there are many files with name pattern {id}-e{num}.ckpt
    get the filename  with max num
    """
    max_num = -1
    latest_file = ""
    for filename in os.listdir(dir):
        if filename.startswith(f"{id}-e") and filename.endswith(".ckpt"):
            try:
                num = int(filename.split("-e")[1].split(".")[0])
                if num > max_num:
                    max_num = num
                    latest_file = filename
            except ValueError:
                continue
    return latest_file


def get_latest_Sovits_weights(dir: str, id: str):
    """
    we have a folder path is dir
    there are many files with name pattern {id}_e{num}_s{digits}.pth
    get the filename  with max num
    """
    max_num = -1
    latest_file = ""
    for filename in os.listdir(dir):
        if filename.startswith(f"{id}_e") and filename.endswith(".pth"):
            try:
                num = int(filename.split("_e")[1].split("_s")[0])
                if num > max_num:
                    max_num = num
                    latest_file = filename
            except ValueError:
                continue
    return latest_file


def set_default():
    global default_batch_size, default_max_batch_size, gpu_info, default_sovits_epoch, default_sovits_save_every_epoch, max_sovits_epoch, max_sovits_save_every_epoch, default_batch_size_s1, if_force_ckpt
    if_force_ckpt = False
    gpu_info = "\n".join(gpu_infos)
    if is_gpu_ok:
        minmem = min(mem)
        default_batch_size = minmem // 2 if version not in v3v4set else minmem // 8
        default_batch_size_s1 = minmem // 2
    else:
        default_batch_size = default_batch_size_s1 = int(
            psutil.virtual_memory().total / 1024 / 1024 / 1024 / 4
        )
    if version not in v3v4set:
        default_sovits_epoch = 8
        default_sovits_save_every_epoch = 4
        max_sovits_epoch = 25  # 40
        max_sovits_save_every_epoch = 25  # 10
    else:
        default_sovits_epoch = 2
        default_sovits_save_every_epoch = 1
        max_sovits_epoch = 16  # 40 # 3 #训太多=作死
        max_sovits_save_every_epoch = 10  # 10 # 3

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3


set_default()

gpus = "-".join(map(str, GPU_INDEX))
default_gpu_numbers = infer_device.index


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if int(input) not in set_gpu_numbers:
            return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


from config import pretrained_gpt_name, pretrained_sovits_name


def check_pretrained_is_exist(version):
    pretrained_model_list = (
        pretrained_sovits_name[version],
        pretrained_sovits_name[version].replace("s2G", "s2D"),
        pretrained_gpt_name[version],
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    )
    _ = ""
    for i in pretrained_model_list:
        if "s2Dv3" not in i and "s2Dv4" not in i and os.path.exists(i) == False:
            _ += f"\n    {i}"
    if _:
        print("warning: ", i18n("以下模型不存在:") + _)


check_pretrained_is_exist(version)
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) == False:
        pretrained_sovits_name[key] = ""
for key in pretrained_gpt_name.keys():
    if os.path.exists(pretrained_gpt_name[key]) == False:
        pretrained_gpt_name[key] = ""

from config import (
    GPT_weight_root,
    GPT_weight_version2root,
    SoVITS_weight_root,
    SoVITS_weight_version2root,
    change_choices,
    get_weights_names,
    bert_path,
    cnhubert_path,
)

for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()


class Transcription(BaseModel):
    start: float
    end: Optional[float] = None
    text: str
    speaker: Optional[str] = None

    def __repr__(self) -> str:
        return f"Transcription(start={self.start}, end={self.end}, text='{self.text}', speaker='{self.speaker}')"

    def __str__(self) -> str:
        return f"[{self.start:.3f}s - {self.end:.3f}s](Speaker: {self.speaker}) {self.text}"

    def to_dict(self) -> dict:
        """Convert Transcription object to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcription":
        """Create Transcription object from dictionary."""
        return cls(**data)


class Guide(BaseModel):
    audio_path: str
    text: str


class ProcessingRequest(BaseModel):
    id: str
    transcriptions: List[Transcription]
    guide: Dict[str, Guide]
    audio: str


class ProcessingResponse(BaseModel):
    success: bool
    message: str


speakers: Dict[str, Guide] = {}


def save_uploaded_files(
    audio_files: List[UploadFile], temp_dir: str, guide: Dict[str, Guide]
) -> List[str]:
    """
    Save uploaded audio files to the temporary directory

    Args:
        audio_files: List of uploaded audio files
        temp_dir: Temporary directory path

    Returns:
        List of saved file paths
    """
    saved_files = []
    global speakers
    speakers = {}
    for i, audio_file in enumerate(audio_files):
        file_path = os.path.join(temp_dir, audio_file.filename)
        # get basename of filename without suffix
        basename = os.path.splitext(os.path.basename(audio_file.filename))[0]
        print(f"Processing file {i + 1}/{len(audio_files)}: {basename}")
        text = guide.get(basename, {}).text
        if not text:
            print(f"Missing text for audio file {audio_file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing text for audio file {audio_file.filename}",
            )
        speakers[basename] = Guide(
            audio_path=file_path,
            text=text,
        )
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            saved_files.append(file_path)

        except Exception as e:
            print(f"Error processing file {audio_file.filename}: {e}")
            # Remove the file if it was partially created
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio file {audio_file.filename}: {str(e)}",
            )

    return saved_files


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        kill_proc_tree(pid)
    print(process_name + i18n("进程已终止"))


def process_info(process_name="", indicator=""):
    if indicator == "opened":
        return process_name + i18n("已开启")
    elif indicator == "open":
        return i18n("开启") + process_name
    elif indicator == "closed":
        return process_name + i18n("已关闭")
    elif indicator == "close":
        return i18n("关闭") + process_name
    elif indicator == "running":
        return process_name + i18n("运行中")
    elif indicator == "occupy":
        return process_name + i18n("占用中") + "," + i18n("需先终止才能开启下一次任务")
    elif indicator == "finish":
        return process_name + i18n("已完成")
    elif indicator == "failed":
        return process_name + i18n("失败")
    elif indicator == "info":
        return process_name + i18n("进程输出信息")
    else:
        return process_name


process_name_slice = "语音切分"
from tools import my_utils
from tools.my_utils import check_details, check_for_existance


def open_slice(
    inp,
    opt_root,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    _max,
    alpha,
    n_parts,
):
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) == False:
        print("输入路径不存在")
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        print("输入路径存在但不可用")
        return

    for i_part in range(n_parts):
        cmd = [
            python_exec,
            "-s",
            "tools/slice_audio.py",
            inp,
            opt_root,
            str(threshold),
            str(min_length),
            str(min_interval),
            str(hop_size),
            str(max_sil_kept),
            str(_max),
            str(alpha),
            str(i_part),
            str(n_parts),
        ]
        subprocess.run(
            cmd,
            cwd="/home/roby/proj/GPT-SoVITS",
            capture_output=True,
            text=True,
            check=True,
        )

        print(process_info(process_name_slice, "opened")),

        # for p in ps_slice:
        #     p.wait()
        print(process_info(process_name_slice, "finish"))
    else:
        print(process_info(process_name_slice, "occupy"))


from tools.asr.config import asr_dict

process_name_asr = i18n("语音识别")


def open_asr(
    asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision
):
    asr_inp_dir = my_utils.clean_path(asr_inp_dir)
    asr_opt_dir = my_utils.clean_path(asr_opt_dir)
    python_exec = "/home/roby/miniconda3/envs/GPTSoVits/bin/python"

    check_for_existance([asr_inp_dir])

    cmd = [
        python_exec,
        "-s",
        f"tools/asr/{asr_dict[asr_model]['path']}",
        "-i",
        asr_inp_dir,
        "-o",
        asr_opt_dir,
        "-s",
        str(asr_model_size),
        "-l",
        str(asr_lang),
        "-p",
        str(asr_precision),
    ]

    print(process_info(process_name_asr, "opened"))

    subprocess.run(
        cmd,
        cwd="/home/roby/proj/GPT-SoVITS",
        capture_output=True,
        text=True,
        check=True,
    )

    print(process_info(process_name_asr, "finish"))


process_name_1a = i18n("文本分词与特征提取")


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    opt_dir = "%s/%s" % (exp_root, exp_name)
    config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": bert_pretrained_dir,
    }
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                "is_half": str(is_half),
            }
        )
        os.environ.update(config)
        print(config)
        env = os.environ.copy()
        cmd = [
            python_exec,
            "-s",
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
        ]

        subprocess.run(
            cmd,
            cwd="/home/roby/proj/GPT-SoVITS",
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    print(process_info(process_name_1a, "running"))

    opt = []
    for i_part in range(all_parts):
        txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    path_text = "%s/2-name2text.txt" % opt_dir
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    if len("".join(opt)) > 0:
        print(process_info(process_name_1a, "finish"))
    else:
        print(process_info(process_name_1a, "failed"))


sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
ps1b = []
process_name_1b = i18n("语音自监督特征提取")


def open1b(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1b == []:
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "sv_path": sv_path,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = (
                '"%s" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
                % python_exec
            )
            print(cmd)
            p = subprocess.Popen(cmd, shell=True)
            ps1b.append(p)
        print(process_info(process_name_1b, "running"))
        for p in ps1b:
            p.wait()
        ps1b = []
        if "Pro" in version:
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" -s GPT_SoVITS/prepare_datasets/2-get-sv.py' % python_exec
                print(cmd)
                p = subprocess.Popen(cmd, shell=True)
                ps1b.append(p)
            for p in ps1b:
                p.wait()
            ps1b = []
        print(process_info(process_name_1b, "finish"))
    else:
        print(process_info(process_name_1b, "occupy"))


def close1b():
    global ps1b
    if ps1b != []:
        for p1b in ps1b:
            try:
                kill_process(p1b.pid, process_name_1b)
            except:
                traceback.print_exc()
        ps1b = []
    return (
        process_info(process_name_1b, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


ps1c = []
process_name_1c = i18n("语义Token提取")


def open1c(version, inp_text, inp_wav_dir, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)
    exp_name = exp_name.rstrip(" ")
    if ps1c == []:
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": config_file,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = subprocess.Popen(cmd, shell=True)
            ps1c.append(p)
        print(process_info(process_name_1c, "running"))
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        print(process_info(process_name_1c, "finish"))
    else:
        print(process_info(process_name_1c, "occupy"))


def close1c():
    global ps1c
    if ps1c != []:
        for p1c in ps1c:
            try:
                kill_process(p1c.pid, process_name_1c)
            except:
                traceback.print_exc()
        ps1c = []
    return (
        process_info(process_name_1c, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_SoVITS = None
process_name_sovits = i18n("SoVITS训练")


def open1Ba(
    version,
    batch_size,
    total_epoch,
    exp_name,
    text_low_lr_rate,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers1Ba,
    pretrained_s2G,
    pretrained_s2D,
    if_grad_ckpt,
    lora_rank,
):
    global p_train_SoVITS
    if p_train_SoVITS == None:
        exp_name = exp_name.rstrip(" ")
        config_file = (
            "GPT_SoVITS/configs/s2.json"
            if version not in {"v2Pro", "v2ProPlus"}
            else f"GPT_SoVITS/configs/s2{version}.json"
        )
        with open(config_file) as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2_%s" % (s2_dir, version), exist_ok=True)
        if check_for_existance([s2_dir], is_train=True):
            check_details([s2_dir], is_train=True)
        if is_half == False:
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)

        save_weight_dir = os.path.join(exp_root, SoVITS_weight_version2root[version])
        os.makedirs(save_weight_dir, exist_ok=True)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["train"]["grad_ckpt"] = if_grad_ckpt
        data["train"]["lora_rank"] = lora_rank
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = save_weight_dir
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))
        if version in ["v1", "v2", "v2Pro", "v2ProPlus"]:
            cmd = '"%s" -s GPT_SoVITS/s2_train.py --config "%s"' % (
                python_exec,
                tmp_config_path,
            )
        else:
            cmd = '"%s" -s GPT_SoVITS/s2_train_v3_lora.py --config "%s"' % (
                python_exec,
                tmp_config_path,
            )
        print(process_info(process_name_sovits, "opened"))
        print(cmd)
        p_train_SoVITS = subprocess.Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        print(process_info(process_name_sovits, "finish"))
    else:
        print(process_info(process_name_sovits, "occupy"))


def close1Ba():
    global p_train_SoVITS
    if p_train_SoVITS is not None:
        kill_process(p_train_SoVITS.pid, process_name_sovits)
        p_train_SoVITS = None
    return (
        process_info(process_name_sovits, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


p_train_GPT = None
process_name_gpt = i18n("GPT训练")


def open1Bb(
    batch_size,
    total_epoch,
    exp_name,
    if_dpo,
    if_save_latest,
    if_save_every_weights,
    save_every_epoch,
    gpu_numbers,
    pretrained_s1,
):
    global p_train_GPT
    if p_train_GPT == None:
        exp_name = exp_name.rstrip(" ")
        with open(
            "GPT_SoVITS/configs/s1longer.yaml"
            if version == "v1"
            else "GPT_SoVITS/configs/s1longer-v2.yaml"
        ) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if check_for_existance([s1_dir], is_train=True):
            check_details([s1_dir], is_train=True)

        half_weights_save_dir = os.path.join(exp_root, GPT_weight_version2root[version])
        os.makedirs(half_weights_save_dir, exist_ok=True)
        if is_half == False:
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = half_weights_save_dir
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(
            gpu_numbers.replace("-", ",")
        )
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" -s GPT_SoVITS/s1_train.py --config_file "%s" ' % (
            python_exec,
            tmp_config_path,
        )
        print(process_info(process_name_gpt, "opened"))
        print(cmd)
        p_train_GPT = subprocess.Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        print(process_info(process_name_gpt, "finish"))
    else:
        print(process_info(process_name_gpt, "occupy"))


def close1Bb():
    global p_train_GPT
    if p_train_GPT is not None:
        kill_process(p_train_GPT.pid, process_name_gpt)
        p_train_GPT = None
    return (
        process_info(process_name_gpt, "closed"),
        {"__type__": "update", "visible": True},
        {"__type__": "update", "visible": False},
    )


@router.post("/tts/", response_model=ProcessingResponse)
async def tts(
    request: Request,
    data: str = Form(
        ..., description="JSON string containing id, transcriptions, and guide"
    ),
    audio_files: List[UploadFile] = File(..., description="Multiple audio files"),
    vocals: UploadFile = File(..., description="Vocals audio file"),
):
    """
    Process TTS request with JSON data and audio files.

    Args:
        data: JSON string containing processing data
        audio_files: List of audio files to process
        vocals: Vocals audio file

    Returns:
        ProcessingResponse with details about processed files
    """
    # Debug logging
    logger.info("=== TTS REQUEST DEBUG ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Content-Type: {request.headers.get('content-type')}")

    # Log form data
    logger.info(f"Raw data received: {data}")
    logger.info(f"Number of audio files: {len(audio_files) if audio_files else 0}")
    logger.info(
        f"Audio files names: {[f.filename for f in audio_files] if audio_files else []}"
    )
    logger.info(f"Vocals file: {vocals.filename if vocals else 'None'}")

    try:
        # Parse JSON data
        try:
            logger.info("Attempting to parse JSON data...")
            json_data = json.loads(data)
            logger.info(f"Parsed JSON successfully: {json_data}")

            # Check required fields before creating ProcessingRequest
            if "id" not in json_data:
                raise ValueError("Missing required field: 'id'")
            if "transcriptions" not in json_data:
                raise ValueError("Missing required field: 'transcriptions'")
            if "guide" not in json_data:
                raise ValueError("Missing required field: 'guide'")
            if "audio" not in json_data:
                raise ValueError("Missing required field: 'audio'")

            logger.info("Creating ProcessingRequest object...")
            request_data = ProcessingRequest(**json_data)
            logger.info(
                f"ProcessingRequest created successfully for ID: {request_data.id}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Raw data that failed to parse: {repr(data)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            logger.error(f"JSON data structure: {json_data}")
            raise HTTPException(
                status_code=400, detail=f"Invalid data structure: {str(e)}"
            )

        # Validate that we have audio files
        if not audio_files:
            logger.error("No audio files provided")
            raise HTTPException(status_code=400, detail="No audio files provided")

        # Create temporary directory for this request
        temp_dir = f"fastapi_audio_processor/tmp/{request_data.id}"
        logger.info(f"Creating temporary directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)

        # Save audio files
        logger.info("Saving audio files...")
        print(request_data.guide)
        saved_audio_files = save_uploaded_files(
            audio_files, temp_dir, request_data.guide
        )
        logger.info(f"Saved {len(saved_audio_files)} audio files")

        # Save vocals file
        logger.info("Saving vocals file...")
        vocals_path = os.path.join(temp_dir, vocals.filename)
        with open(vocals_path, "wb") as buffer:
            shutil.copyfileobj(vocals.file, buffer)
        logger.info(f"Vocals file saved to: {vocals_path}")

        # Create processing summary
        summary = {
            "id": request_data.id,
            "timestamp": datetime.now().isoformat(),
            "temp_dir": temp_dir,
            "audio_files_count": len(saved_audio_files),
            "audio_files": [os.path.abspath(f) for f in saved_audio_files],
            "vocals_file": os.path.abspath(vocals_path),
            "transcription_count": (
                len(request_data.transcriptions) if request_data.transcriptions else 0
            ),
        }

        # Save summary to file
        summary_file = os.path.join(temp_dir, "processing_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Request processed successfully")

        global exp_root
        exp_root = os.path.join(temp_dir, "logs")
        # step 1 slice vocals
        logger.info("=== STEP 1: Slicing vocals file ===")
        try:
            slice_output_dir = os.path.join(temp_dir, "slicer_opt")
            os.makedirs(slice_output_dir, exist_ok=True)
            open_slice(
                vocals_path, slice_output_dir, -34, 5000, 500, 10, 5000, 0.9, 0.25, 4
            )

        except Exception as e:
            logger.error(f"Step 1 failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Audio slicing failed: {str(e)}"
            )

        # STEP 2: ASR using whisper
        logger.info("=== STEP 2: Running ASR with Whisper ===")
        try:

            asr_output_dir = os.path.join(temp_dir, "asr_opt")
            os.makedirs(asr_output_dir, exist_ok=True)
            open_asr(
                slice_output_dir,
                asr_output_dir,
                "Faster Whisper (多语种)",
                "large-v3",
                "en",
                "float16",
            )

        except Exception as e:
            logger.error(f"Step 2 failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")

        inp_text = os.path.join(asr_output_dir, "slicer_opt.list")

        # STEP 3: Get text  (1Aa)
        logger.info("=== STEP 3: Getting text ===")
        try:
            open1a(
                inp_text,
                slice_output_dir,
                request_data.id,
                "0-0",
                bert_path,
            )
            logger.info("Get text completed successfully")

        except Exception as e:
            logger.error(f"Step 3 failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Get text failed: {str(e)}")

        # STEP 4: Get hubert wav32k(1Ab)
        logger.info("=== STEP 4: Getting hubert wav32k ===")
        try:

            open1b(
                version,
                inp_text,
                slice_output_dir,
                request_data.id,
                "0-0",
                cnhubert_path,
            )
            logger.info("Get hubert wav32k completed successfully")

        except Exception as e:
            logger.error(f"Step 4 failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Get hubert wav32k failed: {str(e)}"
            )

        # STEP 5: Get semantic(1Ac)
        logger.info("=== STEP 6: Getting semantic ===")
        try:
            open1c(
                version,
                inp_text,
                slice_output_dir,
                request_data.id,
                "0-0",
                pretrained_sovits_name[version],
            )
            logger.info("Get semantic completed successfully")

        except Exception as e:
            logger.error(f"Step 6 failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Get semantic failed: {str(e)}"
            )

        global tmp
        tmp = os.path.join(temp_dir, "TEMP")
        os.makedirs(tmp, exist_ok=True)
        os.environ["TEMP"] = tmp
        if os.path.exists(tmp):
            for name in os.listdir(tmp):
                if name == "jieba.cache":
                    continue
                path = "%s/%s" % (tmp, name)
                delete = os.remove if os.path.isfile(path) else shutil.rmtree
                try:
                    delete(path)
                except Exception as e:
                    print(str(e))
                    pass

        # STEP 6: Train s2
        logger.info("=== STEP 7: Training s2 ===")
        try:
            open1Ba(
                version,
                4,
                12,
                request_data.id,
                0.4,
                True,
                True,
                12,
                "0",
                pretrained_sovits_name[version],
                pretrained_sovits_name[version].replace("s2G", "s2D"),
                False,
                32,
            )
            logger.info("S2 training completed successfully")

        except Exception as e:
            logger.error(f"Step 7 failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"S2 training failed: {str(e)}")

        # STEP 7: Train s1
        logger.info("=== STEP 8: Training s1 ===")
        try:
            open1Bb(
                4,
                12,
                request_data.id,
                False,
                True,
                True,
                12,
                "0",
                pretrained_gpt_name[version],
            )
            logger.info("S1 training completed successfully")

        except Exception as e:
            logger.error(f"Step 8 failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"S1 training failed: {str(e)}")

        # step 8 reference
        logger.info("=== STEP 9: reference ===")

        gpt_dir = os.path.join(temp_dir, "logs/GPT_weights_v2Pro")
        gpt_name = get_latest_gpt_weights(gpt_dir, request_data.id)
        gpt_path = os.path.join(gpt_dir, gpt_name)

        sovits_dir = os.path.join(temp_dir, "logs/SoVITS_weights_v2Pro")
        sovits_name = get_latest_Sovits_weights(sovits_dir, request_data.id)
        sovits_path = os.path.join(sovits_dir, sovits_name)

        outputs = []

        for i, t in enumerate(request_data.transcriptions):
            target_text = t.text.strip()
            if not target_text:
                logger.warning(f"Skipping transcription {i} with empty text")
                continue

            target_speaker = t.speaker
            if not target_speaker:
                logger.warning(f"Skipping transcription {i} with empty speaker")
                continue
            reference_audio_path = speakers.get(target_speaker, {}).audio_path
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                logger.warning(
                    f"Skipping transcription {i} with missing or invalid reference audio for speaker '{target_speaker}'"
                )
                continue
            reference_text = speakers.get(target_speaker, {}).text
            if not reference_text:
                logger.warning(
                    f"Skipping transcription {i} with missing reference text for speaker '{target_speaker}'"
                )
                continue

            output_dir = os.path.join(temp_dir, "output")
            filename = f"t_{i}_0.wav"
            os.makedirs(output_dir, exist_ok=True)
            inference(
                gpt_path,
                sovits_path,
                reference_audio_path,
                reference_text,
                i18n("英文"),
                target_text,
                i18n("中文"),
                output_dir,
                filename,
            )
            output_path = os.path.join(output_dir, filename)
            outputs.append(output_path)
        abs_outputs = [os.path.abspath(f) for f in outputs]
        # Create processing summary
        summary = {
            "id": request_data.id,
            "timestamp": datetime.now().isoformat(),
            "transcription_count": len(request_data.transcriptions),
            "audio_files_count": len(saved_audio_files),
            "audio_files": abs_outputs,
            "source": "local",  # [TODO] add multiple source
        }

        # Convert summary to string for response
        summary_str = json.dumps(summary, indent=2, ensure_ascii=False)

        # Save summary to a file
        summary_file = os.path.join(temp_dir, "processing_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_str)
        print(f"Processing summary saved to {summary_file}")

        return ProcessingResponse(
            success=True,
            message=summary_str,
        )

    except HTTPException:
        logger.error("HTTPException occurred during processing", exc_info=True)
        raise HTTPException(status_code=400, detail="Bad request")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
