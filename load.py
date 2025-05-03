"""
Code to load the features
"""

import pickle
import numpy as np
import os

data4_dir, data4_dir_matrix = 'Data4/transcription_areas.pkl', 'Data4/distance_matrices.npz'

def list_available_data():
    """
    列出可用的数据集及其包含的特征
    """
    available_data = {
        'Data4': {
            'description': '中国语言资源保护工程',
            'features': [
                'word_names', 'areas', 'slice', 'slices', 'coords',
                'initials', 'finals', 'tones',
                'initials_distance', 'finals_distance', 'tones_distance', 'overall_distance'
            ]
        }
        # 可以添加其他数据集
    }
    return available_data

def load_feats(name, features=None):
    """
    加载指定数据集的指定特征

    Args:
        name (str): 数据集名称 (e.g., 'Data4')
        features (list, optional): 需要加载的特征列表.
                                     如果为None，则加载所有特征. Defaults to None.

    Returns:
        dict: 包含请求特征的字典
    """
    loaded_data = {}

    if name == 'Data4':
        all_features = [
            'word_names', 'areas', 'slice', 'slices', 'coords',
            'initials', 'finals', 'tones',
            'initials_distance', 'finals_distance', 'tones_distance', 'overall_distance'
        ]
        if features is None:
            features_to_load = all_features
        else:
            features_to_load = [f for f in features if f in all_features]
            if len(features_to_load) != len(features):
                print(f"Warning: Some requested features for {name} are not available.")

        # 加载 pkl 文件中的数据
        pkl_features = ['word_names', 'areas', 'slice', 'slices', 'coords', 'initials', 'finals', 'tones']
        if any(f in features_to_load for f in pkl_features):
            try:
                with open(data4_dir, 'rb') as f:
                    data_dict = pickle.load(f)
                if 'initial' in features_to_load:
                     loaded_data['initials'] = data_dict.get('initial')
                if 'final' in features_to_load:
                     loaded_data['finals'] = data_dict.get('final')
                if 'tone' in features_to_load:
                     loaded_data['tones'] = data_dict.get('tone')
                if 'word_names' in features_to_load:
                    loaded_data['word_names'] = data_dict.get('word_name')
                if 'areas' in features_to_load:
                    loaded_data['areas'] = data_dict.get('area')
                if 'slice' in features_to_load:
                    loaded_data['slice'] = data_dict.get('slice')
                if 'slices' in features_to_load:
                    loaded_data['slices'] = data_dict.get('slices')
                if 'coords' in features_to_load:
                    loaded_data['coords'] = data_dict.get('coords')

            except FileNotFoundError:
                print(f"Error: {data4_dir} not found.")
            except Exception as e:
                print(f"Error loading {data4_dir}: {e}")


        # 加载 npz 文件中的数据
        npz_features = ['initials_distance', 'finals_distance', 'tones_distance', 'overall_distance']
        if any(f in features_to_load for f in npz_features):
            try:
                loaded_npz = np.load(data4_dir_matrix)
                if 'initials_distance' in features_to_load:
                    loaded_data['initials_distance'] = loaded_npz.get('initials')
                if 'finals_distance' in features_to_load:
                    loaded_data['finals_distance'] = loaded_npz.get('finals')
                if 'tones_distance' in features_to_load:
                    loaded_data['tones_distance'] = loaded_npz.get('tones')
                if 'overall_distance' in features_to_load:
                     loaded_data['overall_distance'] = loaded_npz.get('overall')
                loaded_npz.close() # Close the npz file

            except FileNotFoundError:
                print(f"Error: {data4_dir_matrix} not found.")
            except Exception as e:
                print(f"Error loading {data4_dir_matrix}: {e}")

    else:
        print(f"Error: Dataset '{name}' not found.")

    return loaded_data