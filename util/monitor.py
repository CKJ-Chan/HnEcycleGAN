import psutil
import GPUtil

def monitor_resources():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} ({gpu.name}) — Load: {gpu.load*100:.1f}%, Mem: {gpu.memoryUtil*100:.1f}%, Temp: {gpu.temperature}°C")
    except Exception as e:
        print("GPU info error:", e)
    
    print(f"CPU: {cpu:.1f}% | RAM: {mem:.1f}%")
