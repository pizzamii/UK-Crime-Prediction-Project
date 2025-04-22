import os
import subprocess
import time
import argparse
import sys
from datetime import datetime

def run_command(command, step_name):
    print("\n" + "="*80)
    print(f"步骤{step_name}")
    print("="*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
        
        if result.stderr:
            print("警告/错误输出:")
            print(result.stderr)
            
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n{command} 成功完成! 耗时: {duration:.2f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {command} 运行失败")
        print(f"错误信息:\n{e.stderr}")
        return False

def run_streamlit(app_script):
    print(f"\n{'='*80}\n启动可视化应用\n{'='*80}\n")
    
    streamlit_process = subprocess.Popen(
        ['streamlit', 'run', app_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(3)
    
    if streamlit_process.poll() is not None:
        _, stderr = streamlit_process.communicate()
        print(f"启动Streamlit应用失败: {stderr}")
        return False
    
    print("Streamlit应用已成功启动!")
    return True

def main():
    print("\n犯罪预测系统启动时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    raw_data = "cleaned_street_data.csv"
    processed_data = "processed_crime_data.csv"
    featured_data = "featured_crime_data.csv"
    target_type = "total_crimes"
    save_to_mysql = True
    
    if not os.path.exists(raw_data):
        print(f"错误: 原始数据文件 {raw_data} 不存在")
        sys.exit(1)
    
    if not run_command(f"python data_preprocessing.py --input {raw_data} --output {processed_data}", "1: 数据预处理"):
        print("错误: 数据预处理失败，停止执行")
        sys.exit(1)
    
    feature_cmd = f"python feature_engineering.py --input {processed_data} --output {featured_data}"
    if save_to_mysql:
        feature_cmd += " --save-to-mysql"
    
    if not run_command(feature_cmd, "2: 特征工程"):
        print("错误: 特征工程失败，停止执行")
        sys.exit(1)
    
    if not run_command(f"python model_training.py --input {featured_data} --target {target_type}", "3: 模型训练"):
        print("错误: 模型训练失败，停止执行")
        sys.exit(1)
    
    if not run_command(f"python crime_prediction.py --input {featured_data} --target {target_type}", "4: 犯罪预测"):
        print("错误: 犯罪预测失败，停止执行")
        sys.exit(1)
    
    if os.path.exists("visualization_app.py"):
        print("\n" + "="*80)
        print("步骤5: 可视化 (可选)")
        print("="*80)
        print("可视化应用可以通过以下命令启动:")
        print(f"    streamlit run visualization_app.py")
        print("可视化应用将在浏览器中打开，显示预测结果和数据分析")
    
    print("\n" + "="*80)
    print("所有步骤已完成!")
    print("="*80)
    print("\n犯罪预测系统执行完成时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main() 