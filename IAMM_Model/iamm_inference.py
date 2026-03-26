import os
import pickle
import sys
from pathlib import Path

import nltk
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# src.utils.config 在 import 时会 parse_args；对 uvicorn 参数做隔离，避免启动时报 unrecognized arguments
_raw_argv = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    from src.models.IAMM.model import IAMM
    from src.utils import config  # noqa: E402
finally:
    sys.argv = _raw_argv


def _ensure_nltk_resources():
    """
    loader.py 在导入时会读取 stopwords；这里提前确保基础 NLTK 资源存在。
    """
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for full_name, short_name in resources:
        try:
            nltk.data.find(full_name)
        except LookupError:
            nltk.download(short_name, quiet=True)


_ensure_nltk_resources()

from src.utils.comet import Comet
from src.utils.constants import EMO_MAP
from src.utils.data.loader import Dataset, collate_fn, process_sent


class IAMMGenerator:
    def __init__(self, vocab_path=None, model_weight_path=None, comet_model_path=None):
        """
        初始化模型，这个函数只会在系统启动时运行一次。
        """
        print("====== 正在初始化 IAMM 共情生成模块 ======")
        self.device = config.device
        self.relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]

        # 对齐训练时超参数，避免 checkpoint 加载 shape mismatch。
        config.ctx_topk = 15
        config.con_topk = 5
        config.cs_topk = 5
        config.depth = 40
        config.heads = 2
        config.hidden_dim = 300
        config.emb_dim = 300
        config.model = "iamm"
        config.pointer_gen = True

        # 1. 加载词表（原项目主要从 dataset_preproc.p 中读取 [data_tra, data_val, data_tst, vocab]）
        if vocab_path is None:
            vocab_path = os.path.join(str(CURRENT_DIR), "data", "ED", "dataset_preproc.p")
        vocab_path = self._resolve_local_path(vocab_path)
        self.vocab = self._load_vocab(vocab_path)

        # 2. 推理权重路径（默认使用你提供的历史训练 checkpoint）
        if model_weight_path is None:
            model_weight_path = os.path.join(
                str(CURRENT_DIR), "save", "test", "IAMM_17599_35.5081_0.5443"
            )
        model_weight_path = self._resolve_local_path(model_weight_path)
        if not os.path.exists(model_weight_path):
            raise FileNotFoundError(f"IAMM 权重文件不存在: {model_weight_path}")

        # 3. 实例化 IAMM 模型
        # decoder_number 在原代码中无实际作用，传 1 即可
        self.model = IAMM(
            vocab=self.vocab,
            decoder_number=1,
            model_file_path=model_weight_path,
            is_eval=True,
            load_optim=False,
        ).to(self.device)

        # 4. 初始化 COMET 模型
        if comet_model_path is None:
            comet_model_path = os.path.join(str(CURRENT_DIR), "data", "Comet")
        comet_model_path = self._resolve_local_path(comet_model_path)
        if os.path.exists(comet_model_path):
            self.comet_model = Comet(comet_model_path, self.device)
        else:
            self.comet_model = None
            print(
                f"[WARN] COMET 模型目录不存在: {comet_model_path}，将使用 none 占位常识特征。"
            )

        print("====== IAMM 模型加载完成 ======")

    def _resolve_local_path(self, path_str):
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str((CURRENT_DIR / path).resolve())

    def _load_vocab(self, vocab_path):
        """
        兼容原项目常见 vocab 存储格式：
        1) dataset_preproc.p -> [data_tra, data_val, data_tst, vocab]
        2) 直接序列化的 vocab 对象
        3) torch.save 保存的对象（含 vocab 键）
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"词表文件不存在: {vocab_path}")

        try:
            with open(vocab_path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            obj = None

        if isinstance(obj, (list, tuple)) and len(obj) >= 4 and obj[3] is not None:
            maybe_vocab = obj[3]
            if hasattr(maybe_vocab, "word2index") and hasattr(maybe_vocab, "index2word"):
                return maybe_vocab

        if isinstance(obj, dict) and "vocab" in obj:
            maybe_vocab = obj["vocab"]
            if hasattr(maybe_vocab, "word2index") and hasattr(maybe_vocab, "index2word"):
                return maybe_vocab

        if hasattr(obj, "word2index") and hasattr(obj, "index2word"):
            return obj

        maybe_torch_obj = torch.load(vocab_path, map_location="cpu")
        if isinstance(maybe_torch_obj, dict) and "vocab" in maybe_torch_obj:
            return maybe_torch_obj["vocab"]
        if hasattr(maybe_torch_obj, "word2index") and hasattr(maybe_torch_obj, "index2word"):
            return maybe_torch_obj

        raise ValueError(f"无法从 `{vocab_path}` 解析出可用 vocab 对象。")

    def _generate_comet_for_event(self, event_tokens):
        """
        生成单个 event 的 5 种 relation 常识结果，返回格式与原始 Dataset 对齐：
        [intent_res, need_res, want_res, effect_res, react_res]
        每个 relation 都是分词后的句子列表（通常 5 个）。
        """
        event_text = " ".join(event_tokens).strip()
        if not event_text:
            event_text = "none"

        relation_outputs = []
        expected_topk = max(1, int(getattr(config, "cs_topk", 5)))
        if self.comet_model is None:
            # 无 COMET 时也保持与训练期一致的 topk 维度，避免后续索引越界
            return [[["none"] for _ in range(expected_topk)] for _ in self.relations]

        for rel in self.relations:
            generated = self.comet_model.generate(event_text, rel)
            processed = [process_sent(s) for s in generated if str(s).strip()]
            if len(processed) == 0:
                processed = [["none"]]
            if len(processed) < expected_topk:
                processed.extend([processed[-1]] * (expected_topk - len(processed)))
            else:
                processed = processed[:expected_topk]
            relation_outputs.append(processed)
        return relation_outputs

    def generate_reply(self, history: list, emotion_label: str) -> str:
        """
        核心推理函数：接收纯文本列表和情感标签，返回生成的字符串。
        """
        if not history:
            history = ["hello"]
        else:
            history = [str(h) for h in history if str(h).strip()]
            if not history:
                history = ["hello"]

        # --- 第一步：数据预处理与归一化 ---
        emotion = str(emotion_label).strip().lower()
        alias_map = {
            "sadness": "sad",
            "joy": "joyful",
            "happy": "joyful",
            "happiness": "joyful",
            "fear": "afraid",
            "surprise": "surprised",
            "anger": "angry",
            "neutral": "content",
        }
        emotion = alias_map.get(emotion, emotion)
        if emotion not in EMO_MAP:
            emotion = "content"

        ctx_tokens = [process_sent(u) for u in history]
        situation_tokens = process_sent(history[-1])

        # --- 第二步：COMET 推理（TODO 2 & 3）---
        # c(context): 与 Dataset.process_comet_data 的 data_type="c" 对齐，按上下文句子组织
        comet_cxt = [self._generate_comet_for_event(utt) for utt in ctx_tokens]
        # s(situation): 与 data_type="s" 对齐，单条 situation
        comet_sit = self._generate_comet_for_event(situation_tokens)

        # 构造单样本 data_dict，然后复用原始 Dataset + collate_fn 的预处理链路
        single_data = {
            "context": [ctx_tokens],
            "target": [["PAD"]],
            "emotion": [emotion],
            "situation": [situation_tokens],
            "emotion_context": [[]],
            "utt_cs": [comet_cxt[-1] if len(comet_cxt) > 0 else comet_sit],
            "comet_cxt": [comet_cxt],
            "comet_sit": [comet_sit],
        }

        dataset = Dataset(single_data, self.vocab)
        item = dataset[0]
        batch = collate_fn([item])

        # --- 第三步：模型生成与解码 ---
        with torch.no_grad():
            sents_g = self.model.decoder_greedy_batch(batch, max_dec_step=30)
            if sents_g and len(sents_g) > 0 and len(sents_g[0]) > 0:
                final_reply = sents_g[0][0].strip()
            else:
                final_reply = "I understand."

        return final_reply