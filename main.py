import subprocess
import sys
import os
import time

# 获取当前项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, description):
    """
    运行单个脚本并捕获状态
    """
    script_path = os.path.join(PROJECT_ROOT, script_name)

    if not os.path.exists(script_path):
        print(f"严重错误: 找不到文件 {script_name}！请检查项目结构。")
        sys.exit(1)

    print(f"\n" + "=" * 50)
    print(f"[步骤启动] {description} ({script_name})")
    print(f"=" * 50)

    start_time = time.time()

    try:
        # 使用 subprocess 运行脚本，这样可以在主程序中捕获运行状态
        result = subprocess.run([sys.executable, script_path], check=True)

        elapsed_time = time.time() - start_time
        print(f"\n[步骤完成] {script_name} 运行成功！耗时: {elapsed_time:.2f} 秒。\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[步骤失败] {script_name} 运行中途报错退出！请检查上面的错误信息。")
        sys.exit(1)  # 如果这一步失败了，直接终止整个主程序，不再往下运行
    except KeyboardInterrupt:
        print(f"\n[用户中断] 手动停止了程序运行。")
        sys.exit(0)


def main():
    print(f"欢迎使用 基于视觉与单线激光雷达融合的路缘检测系统")
    print(f"工程路径: {PROJECT_ROOT}")
    time.sleep(1)

    total_start = time.time()

    # 按照严格的依赖顺序定义要运行的脚本
    pipeline = [
        ("export_data.py", "1. 视觉感知模块 (YOLOP 语义掩码提取)"),
        ("save_lidar_data.py", "2. 激光雷达模块 (单线模拟与降维投影)"),
        ("fusion.py", "3. 多模态后融合模块 (三重安全锁联合判定)")
    ]

    # 循环执行流水线
    for script, desc in pipeline:
        run_script(script, desc)

    total_elapsed = time.time() - total_start
    print("=" * 50)
    print(f" 全部任务完美竣工！总耗时: {total_elapsed:.2f} 秒。")
    print(f"请前往 output/fusion_results_advanced 文件夹查看最终的融合效果图！")
    print("=" * 50)


if __name__ == "__main__":
    main()