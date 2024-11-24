import platform
import subprocess

os_type = platform.system()

if os_type == "Windows":
    print("Detected Windows OS.")
    process1 = subprocess.Popen(["python", "TorchModel_train.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    process2 = subprocess.Popen(["python", "GPSR_train.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

elif os_type == "Linux":
    print("Detected Linux OS.")
    process1 = subprocess.Popen(["xterm", "-e", "python TorchModel_train.py"])
    process2 = subprocess.Popen(["xterm", "-e", "python GPSR_train.py"])

elif os_type == "Darwin":  # macOS
    print("Detected macOS.")
    process1 = subprocess.Popen(["open", "-a", "Terminal", "--args", "python TorchModel_train.py"])
    process2 = subprocess.Popen(["open", "-a", "Terminal", "--args", "python GPSR_train.py"])

else:
    print(f"Unsupported OS: {os_type}")
    exit(1)

process1.wait()
process2.wait()

print("Both scripts have completed execution.")
