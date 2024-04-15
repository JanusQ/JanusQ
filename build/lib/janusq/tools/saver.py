import logging
import os
import cloudpickle as pickle

def check_and_create_folder(folder_path):
    """
    检查文件夹是否存在，如果不存在则创建该文件夹。

    参数：
    - folder_path: 要检查的文件夹路径

    返回：
    - 无
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError as e:
            logging.warning(e)
            
TEMP_DIR = './temp'

def dump(name, obj):
    check_and_create_folder(TEMP_DIR)
    
    with open(os.path.join(TEMP_DIR, name), 'wb') as f:
        pickle.dump(obj, f)
        
def load(name):
    # check_and_create_folder(TEMP_DIR)
    
    with open(os.path.join(TEMP_DIR, name), 'rb') as f:
        return pickle.load(f)