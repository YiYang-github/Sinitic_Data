"""
Code to load the features
"""

import numpy as np
import pickle
import os # 导入 os 库用于路径拼接


BASE_DATA_DIR = 'Data4' # 数据文件所在的基准目录

# Data4 的原始转写和元数据 pkl 文件
data4_raw_data_path = os.path.join(BASE_DATA_DIR, 'transcription_areas.pkl') 
data4_distance_matrix_path = os.path.join(BASE_DATA_DIR, 'distance_matrices.npz')
data4_processed_info_path = os.path.join(BASE_DATA_DIR, 'processed_info.pkl')
# -------------------------------------------------------------


def load_feats(name, type=None, features=None):
    """
    加载指定数据集的指定类型或指定特征的数据。

    Args:
        name (str): 数据集名称 (e.g., 'Data4')
        type (str, optional): 需要加载的数据类型 (e.g., 'raw', 'distance_matrices').
                              如果指定 type，函数会加载该类型下的预定义特征。
                              如果 type 为 None，则必须通过 features 参数指定要加载的特征。
        features (list, optional): 需要加载的特征列表.
                                     如果 type 已指定，features 可用于过滤该类型下的特征。
                                     如果 type 为 None，则加载此列表中指定的特征。
                                     Defaults to None.

    Returns:
        dict: 包含请求特征的字典，键为特征名，值为对应的数据。
              如果加载失败或未找到数据集/类型，返回空字典或 None。
    """
    loaded_data = {}

    # --- 定义不同数据集和类型下的特征列表和文件路径 ---
    # 这个字典定义了每个数据集名称下，不同类型对应哪些预定义特征以及从哪个文件加载
    DATASET_CONFIG = {
        'Data4': {
            'raw': {
                'file': data4_raw_data_path,
                'pkl_keys': ['word_name', 'area', 'slice', 'slices', 'coords', 'initial', 'final', 'tone'],
                'loader': 'pickle' # 指定加载方式
            },
            'distance_matrices': {
                'file': data4_distance_matrix_path,
                'npz_keys': ['initials', 'finals', 'tones', 'overall'], # npz 文件中的键名
                'output_keys': ['initials_distance', 'finals_distance', 'tones_distance', 'overall_distance'], # 输出字典中的键名
                'loader': 'numpy_npz' # 指定加载方式
            },
            'info': {
                'file': data4_processed_info_path,
                # 处理后信息文件的键名和输出键名一致
                'pkl_keys': ['areas', 'slice', 'slices', 'coords', 'word_names'], # 注意这里的键名与保存时字典的键名对应
                'loader': 'pickle'}
            # 可以继续添加其他 type...
            # 'another_type': {...}
        }
        # 可以继续添加其他 name...
        # 'AnotherDataset': {...}
    }

    # --- 检查数据集名称是否存在 ---
    if name not in DATASET_CONFIG:
        print(f"错误: 数据集 '{name}' 的配置不存在。")
        return {} # 返回空字典表示失败

    dataset_config = DATASET_CONFIG[name]

    # --- 根据 type 或 features 确定要加载的特征和文件 ---
    features_to_load_final = [] # 最终确定要加载的特征列表
    file_to_load = None
    loader_type = None
    source_keys = {} # 记录源文件中的键名到目标输出键名的映射

    if type:
        if type not in dataset_config:
            print(f"错误: 数据集 '{name}' 不支持类型 '{type}'。")
            return {}

        type_config = dataset_config[type]
        file_to_load = type_config.get('file')
        loader_type = type_config.get('loader')

        if loader_type == 'pickle':
            all_type_features = type_config.get('pkl_keys', [])
            # pickle 加载是直接从字典取，源键和输出键一致
            source_keys = {k: k for k in all_type_features}
        elif loader_type == 'numpy_npz':
            all_type_features = type_config.get('output_keys', [])
            # npz 加载需要处理源键到输出键的映射
            npz_keys = type_config.get('npz_keys', [])
            output_keys = type_config.get('output_keys', [])
            if len(npz_keys) == len(output_keys):
                 source_keys = dict(zip(output_keys, npz_keys)) # 存储 输出键 -> 源键 映射
            else:
                 print(f"配置错误: 数据集 '{name}', 类型 '{type}' 的 npz_keys 和 output_keys 数量不匹配。")
                 return {}
        else:
             print(f"配置错误: 数据集 '{name}', 类型 '{type}' 指定了未知的 loader '{loader_type}'。")
             return {}


        if features is None:
            # 如果没有指定 features，则加载该 type 下的所有预定义特征
            features_to_load_final = all_type_features
        else:
            # 如果指定了 features，则加载该 type 下 features 中包含的特征
            features_to_load_final = [f for f in features if f in all_type_features]
            if len(features_to_load_final) != len(features):
                # 检查用户请求的 features 中是否有不属于该 type 的
                not_available = [f for f in features if f not in all_type_features]
                print(f"警告: 请求的特征 {not_available} 不属于数据集 '{name}' 的类型 '{type}'，将被忽略。")

    elif features is not None:
         # 如果 type 为 None 但指定了 features (旧的使用方式，可以保留兼容性或弃用)
         # 为了新设计清晰，建议要求必须指定 type
         print("错误: 未指定数据加载类型 (type)，请指定如 type='raw'。")
         return {}
         # 以下是保留旧 features 用法的代码，如果需要兼容可以启用：
         # print("警告: 未指定数据类型 (type)，尝试按特征列表加载 (旧模式)。")
         # # 需要遍历所有 type 才能找到哪些特征在哪里，比较复杂，不推荐。
         # # 更好的方式是：要求用户必须指定 type。
         # pass # 在新设计中不推荐无 type 加载

    else:
        # type 和 features 都为 None
        print("错误: 既未指定数据加载类型 (type)，也未指定要加载的特征列表 (features)。")
        return {}


    # --- 执行数据加载 ---
    if not file_to_load:
         print("内部错误: 未能确定要加载的文件路径。")
         return {}

    if not features_to_load_final:
        print(f"未找到需要加载的特征列表，请检查 type 或 features 参数。")
        return {}

    print(f"正在从文件 '{file_to_load}' 加载数据...")
    print(f"计划加载的特征: {features_to_load_final}")

    try:
        if loader_type == 'pickle':
            with open(file_to_load, 'rb') as f:
                data_dict = pickle.load(f)

            for feature_name in features_to_load_final:
                # 从加载的字典中按特征名获取数据
                # 使用 .get() 避免 KeyError 如果文件中的确缺少该特征
                if feature_name in data_dict:
                     loaded_data[feature_name] = data_dict[feature_name]
                else:
                     print(f"警告: 在文件 '{file_to_load}' 中未找到特征 '{feature_name}'。")

        elif loader_type == 'numpy_npz':
            loaded_npz = np.load(file_to_load)
            for output_key in features_to_load_final:
                 source_key = source_keys.get(output_key) # 获取 npz 文件中的对应键
                 if source_key and source_key in loaded_npz:
                     loaded_data[output_key] = loaded_npz[source_key]
                 else:
                     print(f"警告: 在文件 '{file_to_load}' 中未找到特征 '{output_key}' (查找键 '{source_key}')。")
            loaded_npz.close() # 关闭 npz 文件句柄

    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_to_load}' 未找到。请检查路径设置。")
        return {} # 返回空字典表示失败
    except Exception as e:
        print(f"错误加载文件 '{file_to_load}': {e}")
        return {} # 返回空字典表示失败

    print(f"成功加载 {len(loaded_data)} 个特征。")
    return loaded_data
