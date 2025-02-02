import psutil

filter_keyword = "/Tools/anaconda3/envs/qto/bin/python"
filter_keyword = "jiangqifan"

# Iterate over all processes
for process in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
    try:
        # Check if process name contains 'python' and matches the filter keyword in the command line
        if process.info['cmdline'] and any(filter_keyword in arg for arg in process.info['cmdline']):
            print(f"Terminating process PID: {process.info['pid']} - {' '.join(process.info['cmdline'])}")
            process.kill()  # Kill the process
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print("All matching processes have been terminated.")
