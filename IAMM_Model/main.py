from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from pathlib import Path
from typing import List
try:
    from iamm_inference import IAMMGenerator
except ModuleNotFoundError:
    # 兼容在仓库根目录使用 `uvicorn IAMM_Model.main:app --reload` 启动
    from IAMM_Model.iamm_inference import IAMMGenerator

# 定义前后端约定的数据传输格式
class ChatRequest(BaseModel):
    history: List[str] = Field(
        ...,
        description="对话历史（按时间顺序）。最后一条通常是用户当前输入。",
        examples=[["I failed my exam today.", "I studied for weeks but still did badly."]],
    )
    current_emotion: str = Field(
        ...,
        description="用户当前情绪标签，例如 sad/joy/neutral。",
        examples=["sad"],
    )

class ChatResponse(BaseModel):
    status: str = Field(..., description="请求状态：success 表示成功。", examples=["success"])
    response: str = Field(..., description="模型生成的共情回复文本。", examples=["I am sorry you are going through this."])


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误信息。")

# 初始化 FastAPI 应用
app = FastAPI(
    title="IAMM 共情对话生成服务",
    version="1.0",
    description=(
        "这是一个基于 IAMM 的共情对话生成 API 服务。\n\n"
        "使用方式：\n"
        "1. 调用 `POST /chat`\n"
        "2. 传入 `history`（对话历史）和 `current_emotion`（当前情绪）\n"
        "3. 服务返回模型生成的共情回复。"
    ),
    openapi_tags=[
        {"name": "对话生成", "description": "与 IAMM 模型交互，生成共情回复。"},
    ],
)
BASE_DIR = Path(__file__).resolve().parent

# 全局变量，用于存储模型实例
ai_generator = None

@app.on_event("startup")
async def startup_event():
    """
    启动服务时自动执行：实例化模型并加载权重。
    请替换为你电脑上真实的权重文件路径。
    """
    global ai_generator
    try:
        # 已替换为当前工程中的真实相对路径
        VOCAB_PATH = "./data/ED/dataset_preproc.p"
        WEIGHT_PATH = "./save/test/IAMM_17599_35.5081_0.5443"

        vocab_abs = BASE_DIR / VOCAB_PATH
        weight_abs = BASE_DIR / WEIGHT_PATH
        if not vocab_abs.exists():
            raise FileNotFoundError(f"词表文件不存在: {vocab_abs}")
        if not weight_abs.exists():
            raise FileNotFoundError(f"模型权重不存在: {weight_abs}")
        
        ai_generator = IAMMGenerator(
            vocab_path=VOCAB_PATH, 
            model_weight_path=WEIGHT_PATH
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise RuntimeError(f"启动失败，模型未就绪: {e}") from e

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["对话生成"],
    summary="生成一条共情回复",
    description=(
        "根据输入的对话历史和当前情绪标签，调用 IAMM 模型进行推理，"
        "返回一条共情风格的回复。"
    ),
    response_description="成功返回模型回复。",
    responses={
        500: {
            "model": ErrorResponse,
            "description": "服务内部错误（例如模型未加载、推理失败等）。",
        }
    },
)
async def generate_chat(request: ChatRequest):
    """
    接收前端对话数据，返回共情回复。
    """
    if not ai_generator:
        raise HTTPException(status_code=500, detail="模型尚未准备好")
        
    try:
        # 调用我们封装好的核心推理类
        reply_text = ai_generator.generate_reply(
            history=request.history,
            emotion_label=request.current_emotion
        )
        
        return ChatResponse(
            status="success",
            response=reply_text
        )
    except Exception as e:
        print(f"生成过程出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 在本地 8000 端口启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)