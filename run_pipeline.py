import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Bắt đầu chạy: {script_path}")
    print(f"{'='*50}\n")
    
    try:
        # Sử dụng sys.executable để lấy đúng đường dẫn Python đang chạy
        result = subprocess.run([sys.executable, script_path], check=True, text=True)
        print(f"\n{'-'*50}")
        print(f"Hoàn thành: {script_path}")
        print(f"{'-'*50}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*50}")
        print(f"LỖI: Chạy {script_path} thất bại với mã lỗi {e.returncode}")
        print(f"{'!'*50}\n")
        return False
    except FileNotFoundError:
        print(f"\n{'!'*50}")
        print(f"LỖI: Không tìm thấy file {script_path}")
        print(f"Đảm bảo bạn đang chạy script này từ thư mục Algorithm_Inno")
        print(f"{'!'*50}\n")
        return False

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY PIPELINE PHÂN TÍCH GIAO DỊCH\n")
    
    # Các bước trong pipeline
    # Đảm bảo đường dẫn này đúng khi đứng ở thư mục gốc của project (Algorithm_Inno)
    step1 = "ML/src/isolation_forest_anomaly_detection.py"
    step2 = "ML/src/random_forest.py"
    
    # Chạy Step 1
    if run_script(step1):
        # Nếu Step 1 thành công, chạy tiếp Step 2
        run_script(step2)
    else:
        print("Pipeline dừng lại vì Step 1 thất bại.")
    
    print("\nKẾT THÚC PIPELINE")
