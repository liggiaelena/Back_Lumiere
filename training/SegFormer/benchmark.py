import torch
import time
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

def run_segformer_benchmark():
    # 網頁 API 常見的輸入解析度 (512x512)
    input_resolution = (512, 512)
    # 模擬一張隨機的臉部影像張量 (Batch_size=1, Channels=3, H=512, W=512)
    dummy_input = torch.randn(1, 3, *input_resolution)
    
    # 檢查是否有 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用測試設備: {device}")
    if device.type == "cuda":
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    print("-" * 70)
    
    dummy_input = dummy_input.to(device)
    variants = ["b0", "b1", "b2", "b3", "b4", "b5"]
    
    print(f"{'模型版本':<15}{'參數大小 (M)':<15}{'平均推論時間 (ms)':<20}{'顯存/記憶體佔用 (MB)':<20}")
    print("-" * 70)
    
    for v in variants:
        model_name = f"nvidia/mit-{v}"
        try:
            # 載入 SegFormer 的 Backbone 架構
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name, 
                num_labels=4 # 預期分類：背景、白斑、黃褐斑、鮮紅斑
            ).to(device)
            model.eval()
            
            # 計算模型總參數集大小 (以百萬為單位)
            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            
            # GPU/CPU 預熱 (Warm-up)，確保正式計時準確
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # 開始進行基準測試（重複執行 50 次取平均）
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(50):
                    _ = model(dummy_input)
                    
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / 50) * 1000
            
            # 記憶體/顯存計算
            if device.type == "cuda":
                mem_used = torch.cuda.memory_allocated(device) / (1024 ** 2)
            else:
                mem_used = 0 # CPU 記憶體通常較難單獨用此 API 精確抓取
                
            print(f"SegFormer-{v:<8}{num_params:<18.2f}{avg_time_ms:<22.2f}{mem_used:<20.2f}")
            
            # 釋放記憶體避免後續模型疊加導致 OOM
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"無法測試 {model_name}，錯誤原因: {e}")

if __name__ == "__main__":
    run_segformer_benchmark()