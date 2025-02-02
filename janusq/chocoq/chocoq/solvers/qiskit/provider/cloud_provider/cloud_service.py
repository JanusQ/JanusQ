from qiskit_ibm_runtime import QiskitRuntimeService
import os
current_dir = os.path.dirname(__file__)

def get_IBM_service(use_free: bool = True, message: str = None, token_idx: int = 0):
    ibm_token_list = []
    with open(os.path.join(current_dir, 'IBM.key'), 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('ibm_token'):
                ibm_token_list.append(line.split('=')[1].strip().strip("'"))
            elif line.startswith('ibm_cloud_api'):
                ibm_cloud_api = line.split('=')[1].strip().strip("'")
            elif line.startswith('ibm_cloud_crn'):
                ibm_cloud_crn = line.split('=')[1].strip().strip("'")
    token_len = len(ibm_token_list)
    try:
        if use_free:
            service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token_list[token_idx % token_len], instance='ibm-q/open/main')
            # 如果这个账号剩余时间小于1分钟，就不要用了
            if service.usage()['byInstance'][0]['usage'] + 60 >= service.usage()['byInstance'][0]['quota']:
                raise Exception(f"token {token_idx % token_len} exhausted")
        else:
            service = QiskitRuntimeService(channel='ibm_cloud', token=ibm_cloud_api, instance=ibm_cloud_crn)
    except Exception as e:
            raise e
    if message:
        print(message)
    return service