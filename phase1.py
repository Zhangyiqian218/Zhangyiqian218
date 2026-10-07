import os
import json
import re
import time
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Pydantic V2 强类型防幻觉装甲
# ==========================================
class PatientProfile(BaseModel):
    age: Union[int, str, None] = None
    gender: Optional[str] = None
    admission_days: Union[int, str, None] = None
    post_op_days: Union[int, str, None] = None
    past_medical_history: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid') 

class MedicationOrder(BaseModel):
    drug_name: str
    intent: List[str]
    indication: Optional[str] = None
    loading_dose: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    course_duration: Optional[str] = None
    adjusted_from_drug: Optional[str] = None
    adjusted_from_dosage: Optional[str] = None
    conditional_actions: Optional[str] = None
    skin_test_required: bool = False
    combination_group: Optional[str] = None
    model_config = ConfigDict(extra='forbid')

class MedicalRecordSchema(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    physical_examination: Dict[str, Any] = Field(default_factory=dict)
    clinical_scores: List[Dict[str, Any]] = Field(default_factory=list)
    diagnoses: List[Dict[str, Any]] = Field(default_factory=list)
    medication_orders: List[MedicationOrder] = Field(default_factory=list)
    fluid_management: List[Dict[str, Any]] = Field(default_factory=list)
    non_pharm_interventions: List[Dict[str, Any]] = Field(default_factory=list)
    exam_management: Dict[str, Any] = Field(default_factory=dict)
    medical_alerts: Dict[str, Any] = Field(default_factory=dict)
    patient_flow_plans: List[Dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid')

# ==========================================
# 系统抽取 Prompt（模块级，供主流程与独立脚本复用）
# ==========================================
SYSTEM_PROMPT = """你是一个顶级的临床医学信息抽取架构师。请严格按照要求抽取，只输出JSON，严禁输出任何分析过程。

    【结构要求】：
    {
      "metadata": {"record_time": "..."},
      "patient_profile": {"age": 0, "gender": "...", "admission_days": 0, "post_op_days": 0, "past_medical_history": []},
      "physical_examination": {"vital_signs": {"temperature": 0.0, "pulse": 0, "respiration": 0, "systolic_bp": 0, "diastolic_bp": 0, "spo2": 0.0}, "abnormal_signs": []},
      "clinical_scores": [{"score_name": "...", "score_value": "...", "interpretation": "..."}],
      "diagnoses": [{"disease_name": "...", "diagnosis_type": "...", "grading": "...", "status": "...", "diagnostic_basis": []}],
      "medication_orders": [{"drug_name": "...", "intent": [], "indication": "...", "loading_dose": "...", "dosage": "...", "frequency": "...", "route": "...", "course_duration": "...", "adjusted_from_drug": "...", "adjusted_from_dosage": "...", "conditional_actions": "...", "skin_test_required": false, "combination_group": "..."}],
      "fluid_management": [{"fluid_type": "...", "total_volume": "...", "infusion_rate": "...", "urine_output_target": "...", "io_balance_goal": "..."}],
      "non_pharm_interventions": [{"intervention_type": "...", "intervention_name": "...", "intent": "...", "details": "..."}],
      "exam_management": {"results": [{"exam_name": "...", "result_value": "...", "unit": "...", "abnormality_flag": "...", "collection_time": "..."}], "orders": [{"exam_name": "...", "urgency": "...", "specimen_type": "...", "conditional_actions": "...", "exclusion_conditions": "..."}]},
      "medical_alerts": {"allergies": [{"allergen": "...", "reaction_type": "...", "severity": "..."}], "contraindications": []},
      "patient_flow_plans": [{"flow_type": "...", "target_department": "...", "purpose": "...", "status_or_timing": "..."}]
    }
    """

# ==========================================
# 2. 核心大模型调度器 (带防抖与自适应配置)
# ==========================================
class StructFlowExtractor:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        # Kimi K2、DeepSeek-R1 等为 reasoning 模型，思考占用大量输出额度，需更高 token/超时
        _name = model_name.lower()
        self.is_reasoning = ("r1" in _name or "reasoning" in _name or "kimi" in _name)
        self.timeout = 150.0 if self.is_reasoning else 45.0
        self.max_tokens = 8000 if self.is_reasoning else 2500
        # Kimi K2 只允许 temperature=1，其余模型用低温保证抽取稳定
        self.temperature = 1.0 if "kimi" in _name else 0.01
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def clean_json_string(self, text: str) -> str:
        if not text:
            return "{}"
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return text[start_idx:end_idx+1]
        return text.strip()

    def evaluate_zero_shot(self, system_prompt: str, user_text: str) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]:
        raw_response = ""
        cleaned_text = ""
        max_attempts = 4 # 1次首发 + 3次指数重试 (1s, 2s, 4s)
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                finish_reason = response.choices[0].finish_reason
                raw_response = response.choices[0].message.content or ""
                error_prefix = "[警告: Token截断] " if finish_reason == "length" else ""

                cleaned_text = self.clean_json_string(raw_response)
                parsed_json = json.loads(cleaned_text)
                MedicalRecordSchema(**parsed_json) 
                
                return raw_response, cleaned_text, parsed_json, f"{error_prefix}修复成功" if error_prefix else None
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return raw_response, cleaned_text, None, f"并发网关崩溃({type(e).__name__}): {str(e)}"
            except json.JSONDecodeError as e:
                return raw_response, cleaned_text, None, f"JSON结构撕裂: {str(e)}"
            except ValidationError as e:
                return raw_response, cleaned_text, None, f"Schema幻觉拦截: {str(e).replace('new lines', ' ')}"
            except Exception as e:
                if "429" in str(e) and attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return raw_response, cleaned_text, None, f"底层异常: {str(e)}"

# ==========================================
# 3. 心理测量学 Phase 1 (严格防错位并发版)
# ==========================================
def run_phase_1_irt_matrix(data_path: str, model_configs: dict, schema_prompt: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "Phase1_01_Matrix.csv")
    logs_path = os.path.join(output_dir, "Phase1_Detailed_Logs.csv")

    df = pd.read_excel(data_path)
    required_columns = {'记录编号', '口述全文'}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"数据源缺少必要列。需包含 {required_columns}")
    
    df['记录编号'] = df['记录编号'].astype(str)

    processed_ids = set()
    if os.path.exists(matrix_path):
        try:
            existing_matrix = pd.read_csv(matrix_path)
            processed_ids = set(existing_matrix['record_id'].astype(str).tolist())
            print(f"🔄 断点保护：跳过 {len(processed_ids)} 条已落盘记录。")
        except Exception as e:
            print(f"⚠️ 历史文件读取异常: {e}")

    extractors = {
        name: StructFlowExtractor(api_key=cfg["api_key"], base_url=cfg["base_url"], model_name=name)
        for name, cfg in model_configs.items() if cfg.get("api_key")
    }
    if not extractors:
        raise ValueError("未找到有效的大模型 API 密钥配置。")

    MATRIX_COLUMNS = ["record_id"] + list(extractors.keys())
    LOG_COLUMNS = ["record_id", "model_name", "score", "latency_sec", "error_type", "cleaned_json", "raw_response"]
    
    records = df.to_dict('records')
    total_records = len(records)
    print(f"✅ 激活 {len(extractors)} 引擎 (含限流防抖与CoT动态适配)...\n" + "-"*50)

    def fetch_model_result(model_name, extractor, user_prompt):
        start_time = time.time()
        raw_text, cleaned_text, parsed_json, error_msg = extractor.evaluate_zero_shot(schema_prompt, user_prompt)
        latency = round(time.time() - start_time, 2)
        return model_name, raw_text, cleaned_text, parsed_json, error_msg, latency

    for idx, row in enumerate(records, 1):
        record_id = str(row.get('记录编号', f'Unknown-{idx}'))
        
        if record_id in processed_ids:
            continue
            
        text = row.get('口述全文', '')
        print(f"▶ [{idx}/{total_records}] 评测病例: {record_id} (并发请求中...)")
        
        context_parts = []
        for key in ['记录日期时间', '年龄', '性别']:
            if key in row and pd.notna(row[key]):
                context_parts.append(f"{key}:{row[key]}")
        
        user_input = f"[辅助上下文]: {' '.join(context_parts)}\n" if context_parts else ""
        user_input += f"[口述全文]:\n{text}"
        
        record_scores = {"record_id": record_id}
        detailed_logs_batch = []
        
        with ThreadPoolExecutor(max_workers=len(extractors)) as executor:
            future_to_model = {
                executor.submit(fetch_model_result, name, ext, user_input): name 
                for name, ext in extractors.items()
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    name, raw_text, cleaned_text, parsed_json, error_msg, latency = future.result()
                    is_success = 1 if parsed_json is not None else 0
                    
                    record_scores[name] = is_success
                    detailed_logs_batch.append({
                        "record_id": record_id, "model_name": name, "score": is_success,
                        "latency_sec": latency, "error_type": error_msg,
                        "cleaned_json": cleaned_text, "raw_response": raw_text
                    })
                except Exception as exc:
                    print(f"  [致命拦截] {model_name} 线程崩溃: {exc}")
                    record_scores[model_name] = 0
                    detailed_logs_batch.append({
                        "record_id": record_id, "model_name": model_name, "score": 0,
                        "latency_sec": 0.0, "error_type": f"线程池执行器崩溃: {str(exc)}",
                        "cleaned_json": "", "raw_response": ""
                    })
        
        for model in extractors.keys():
            if model not in record_scores:
                record_scores[model] = 0

        # 按固定列顺序显式取位置值，杜绝断点续跑时的字典对齐错位
        row_values = [record_id] + [record_scores.get(name, 0) for name in extractors.keys()]
        pd.DataFrame([row_values], columns=MATRIX_COLUMNS).to_csv(
            matrix_path, mode='a', header=not os.path.exists(matrix_path), index=False
        )
        pd.DataFrame(detailed_logs_batch, columns=LOG_COLUMNS).to_csv(
            logs_path, mode='a', header=not os.path.exists(logs_path), index=False, encoding='utf-8-sig'
        )
        
        processed_ids.add(record_id)
        time.sleep(1) 

    print("\n" + "="*50)
    print("🏆 大模型 L0 Schema 零样本抽取排行")
    print("="*50)
    
    if os.path.exists(matrix_path) and os.path.exists(logs_path):
        final_matrix = pd.read_csv(matrix_path)
        final_logs = pd.read_csv(logs_path)

        # 合并单模型补跑结果（kimi_results.csv），用于延迟等统计
        kimi_path = os.path.join(os.path.dirname(logs_path), "kimi_results.csv")
        if os.path.exists(kimi_path):
            kr = pd.read_csv(kimi_path).rename(columns={"kimi-k2.6": "score"})
            kr["model_name"] = "kimi-k2.6"
            keep = [c for c in ["record_id", "model_name", "score", "latency_sec", "error_type"] if c in kr.columns]
            final_logs = pd.concat([final_logs, kr[keep]], ignore_index=True)

        pass_rates = final_matrix.drop(columns=['record_id'], errors='ignore').mean(numeric_only=True).sort_values(ascending=False)
        avg_latency = final_logs.groupby('model_name')['latency_sec'].mean()
        
        rank = 1
        for model, rate in pass_rates.items():
            latency = avg_latency.get(model, 0)
            print(f"{'🏅' if rank<=3 else '🔸'} Rank {rank} | {model:<18} | 准确率: {rate:>6.1%} | 均耗时: {latency:>5.1f}s")
            rank += 1
            
        print(f"\n✅ Phase 1 数据已全量固化至: {output_dir}")

# ==========================================
# 4. 主程序入口配置 (10大异构模型矩阵)
# ==========================================
if __name__ == "__main__":
    load_dotenv()
    
    ACTIVE_MODELS = {
        # --- 阿里百炼阵营 ---
        "qwen-max": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": os.getenv("DASHSCOPE_API_KEY")},
        "qwen-plus": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": os.getenv("DASHSCOPE_API_KEY")},
        "qwen-turbo": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": os.getenv("DASHSCOPE_API_KEY")},
        "deepseek-v3": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": os.getenv("DASHSCOPE_API_KEY")},
        "deepseek-r1": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": os.getenv("DASHSCOPE_API_KEY")},
        
        # --- 智谱阵营 ---
        "glm-4-plus": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": os.getenv("ZHIPU_API_KEY")},
        "glm-4-air": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": os.getenv("ZHIPU_API_KEY")},
        "glm-4-flash": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": os.getenv("ZHIPU_API_KEY")},
        
        # --- Moonshot 阵营（Kimi K2，模型名已更新） ---
        "kimi-k2.6": {"base_url": "https://api.moonshot.cn/v1", "api_key": os.getenv("MOONSHOT_API_KEY")}
    }

    ROOT = Path(__file__).resolve().parent
    try:
        run_phase_1_irt_matrix(
            data_path=str(ROOT / "data" / "medical_rounds_50.xlsx"),
            model_configs=ACTIVE_MODELS,
            schema_prompt=SYSTEM_PROMPT,
            output_dir=str(ROOT / "output_eval")
        )
    except FileNotFoundError as e:
        print(f"\n❌ 文件读取失败: {str(e)}")
    except KeyError as e:
        print(f"\n❌ 数据结构错误: {str(e)}")
    except ValueError as e:
        print(f"\n❌ 配置阻断: {str(e)}")
    except Exception as e:
        print(f"\n❌ 程序遭遇未知致命错误: {str(e)}")