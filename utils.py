import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 导入 font_manager

import os
import typing

def get_cjk_font_prop(font_path):
    """
    尝试加载指定路径的字体文件并返回 FontProperties 对象。
    如果失败则返回 None。
    """
    try:
        font_prop = fm.FontProperties(fname=font_path)
        # 简单检查字体是否可用 (Optional)
        # text = "测试中文"
        # plt.figure()
        # plt.text(0.5, 0.5, text, fontproperties=font_prop)
        # plt.close()
        return font_prop
    except Exception as e:
        print(f"Error loading font from {font_path}: {e}")
        return None


def count_and_plot_categories(data_array, font_path=None, title="类别分布柱状图", xlabel="类别", ylabel="数量"):
    """
    统计一维numpy数组中不同类别的数量，并绘制柱状图。
    使用指定的字体路径来显示中文。

    Args:
        data_array (np.ndarray): 输入的一维numpy数组，包含不同类别的字符串。
        font_path (str, optional): 中文字体文件的路径。如果提供，将使用此字体。Defaults to None.
        title (str): 柱状图的标题。
        xlabel (str): x轴的标签。
        ylabel (str): y轴的标签。

    Returns:
        tuple: 包含两个numpy数组，第一个是唯一类别，第二个是对应类别的数量。
    """
    if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
        print("输入必须是一维numpy数组。")
        return None, None

    font_prop = None
    if font_path:
        font_prop = get_cjk_font_prop(font_path=font_path)
        if font_prop is None:
             print("\n###########################################################")
             print("### WARNING: CJK font was NOT loaded successfully.      ###")
             print("### Please double-check the path below is correct     ###")
             print(f"### Path tried: {font_path}                     ###")
             print("### Chinese characters in plots will likely be boxes. ###")
             print("###########################################################\n")
        else:
            print(f"\n--- Font '{font_prop.get_name()}' loaded successfully. Proceeding with plotting. ---\n")


    # 统计不同类别的数量
    unique_categories, counts = np.unique(data_array, return_counts=True)

    # 按数量降序排序
    sorted_indices = np.argsort(counts)[::-1]
    unique_categories = unique_categories[sorted_indices]
    counts = counts[sorted_indices]

    # 绘制柱状图
    plt.figure(figsize=(10, 6)) # 可以调整图的大小
    plt.bar(unique_categories, counts, color='skyblue')

    # 应用字体属性到图表元素
    if font_prop:
        plt.title(title, fontproperties=font_prop)
        plt.xlabel(xlabel, fontproperties=font_prop)
        plt.ylabel(ylabel, fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop) # 旋转x轴标签，避免重叠

        # 在每个柱子上方显示数量，也使用指定字体
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontproperties=font_prop)
    else:
        # 如果字体加载失败，使用默认字体，可能会显示乱码
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right') # 旋转x轴标签，避免重叠

        # 在每个柱子上方显示数量
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')


    plt.tight_layout() # 调整布局，使标签不被裁剪
    plt.show()

    return unique_categories, counts


import numpy as np
import matplotlib.pyplot as plt
import os
import typing

# 请确保在函数外部定义和导入 FontProperties 如果需要自定义字体的话
# from matplotlib.font_manager import FontProperties


def plot_2d_embedding_improved(X,                   # 2D 数据 (n_samples, 2)
                               y_labels,            # 每个点的标签 (n_samples,)
                               plot_title,          # 图表标题 (字符串)
                               save_path,           # 保存图片的完整路径 (e.g., './plots/pca.png')
                               label_title="区域/标签", # 图例标题
                               font_prop=None,      # 传递获取到的 FontProperties 对象
                               plot_labels=True     # 是否绘制每个点的文本标签，默认为 True
                              ):
    """
    生成并保存降维结果的 2D 散点图 (精简版，支持中文)。
    此版本不包含 adjustText 标签防重叠功能。

    Args:
        X (np.ndarray): 2D NumPy 数组，形状为 (n_samples, 2)。
        y_labels (list or np.ndarray): 每个数据点的标签，一维列表或 NumPy 数组，长度为 n_samples。
        plot_title (str): 图表的主标题。
        save_path (str): 保存图表的完整文件路径 (例如，'./plots/pca.png')。
        label_title (str, optional): 图例的标题。默认为 "区域/标签"。
        font_prop (FontProperties, optional): FontProperties 对象，用于正确显示中文标题、标签和图例。默认为 None。
        plot_labels (bool, optional): 是否在图表中每个点旁边绘制文本标签。如果为 False，则只绘制散点和图例。默认为 True。

    Returns:
        None: 函数不返回任何值，主要副作用是保存图表文件。
    """
    print(f"\n--- Plotting: {plot_title} ---")
    # --- 输入验证 ---
    if X is None:
        print("Error: Input data 'X' is None. Skipping plot.")
        return
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 2:
        print(f"Error: Input data 'X' must be a 2D NumPy array with shape (n_samples, 2). Current shape={X.shape}. Skipping plot.")
        return

    if y_labels is None:
        print("Error: 'y_labels' is None. Skipping plot.")
        return

    # *** y_labels 验证和转换为 1D NumPy 数组 ***
    if not isinstance(y_labels, (list, np.ndarray)):
         print(f"Error: 'y_labels' must be a list or a NumPy array. Current type is {type(y_labels)}. Skipping plot.")
         return

    if isinstance(y_labels, np.ndarray) and y_labels.ndim != 1:
         print(f"Error: 'y_labels' NumPy array must be 1D. Current dimension is {y_labels.ndim}. Skipping plot.")
         return

    if len(y_labels) != X.shape[0]:
        print(f"Error: 'y_labels' length ({len(y_labels)}) does not match number of data points in X ({X.shape[0]}). Skipping plot.")
        return

    if isinstance(y_labels, list):
        y_labels = np.array(y_labels)
        if y_labels.ndim != 1 or y_labels.shape[0] != X.shape[0]:
             print(f"Internal Error: Failed to convert y_labels list to 1D numpy array correctly after validation. Skipping plot.")
             return
    # *** 验证结束 ***


    # --- 确保保存目录存在 ---
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            print(f"Created save directory: {save_dir}")
        except OSError as e:
            print(f"Error creating save directory {save_dir}: {e}. Cannot save plot.")
            return

    # --- 绘图设置 ---
    plt.figure(figsize=(12, 9))
    ax = plt.gca() # Get current axes

    unique_labels = np.unique(y_labels)
    num_unique_labels = len(unique_labels)

    # --- 获取颜色映射 ---
    try:
        cmap = plt.get_cmap('tab20', max(20, num_unique_labels))
    except (ValueError, TypeError) as e:
        print(f"Warning: Colormap 'tab20' issue ({e}). Falling back to 'viridis'.")
        try:
             cmap = plt.get_cmap('viridis', max(20, num_unique_labels))
        except TypeError:
             cmap = plt.get_cmap('viridis')
             print("Warning: Fallback 'viridis' also doesn't support num_unique_labels argument. Using default colors from map.")

    # --- 获取标记类型 ---
    markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', 'h', 'H', '+', 'x', '|', '_'] * (num_unique_labels // 20 + 1)

    handles = [] # 用于图例
    # texts 列表和 adjust_text 相关代码已移除

    # --- 绘制散点和（可选的）文本标签 ---
    for i, name in enumerate(unique_labels):
        indices = np.where(y_labels == name)[0]

        if len(indices) == 0:
             print(f"Warning: Label '{name}' has no corresponding data points in X. Skipping.")
             continue

        # --- 获取颜色 ---
        try:
            color = cmap(i / max(1, num_unique_labels - 1))
        except (TypeError, ValueError):
             if hasattr(cmap, 'colors') and len(cmap.colors) > 0:
                  color = cmap.colors[i % len(cmap.colors)]
             else:
                  print(f"Error: Could not get color for label '{name}' from colormap. Using default scatter color.")
                  color = None


        marker = markers[i % len(markers)]

        # 绘制散点 (always plot points)
        scatter = ax.scatter(X[indices, 0], X[indices, 1],
                           color=color, marker=marker, label=str(name),
                           s=60, alpha=0.8, edgecolors='w', linewidth=0.5)
        handles.append(scatter) # 添加到图例句柄

        # *** 根据 plot_labels 参数决定是否添加文本标签 ***
        if plot_labels:
            for idx in indices:
                label_text = str(y_labels[idx]) # 确保标签是字符串
                text_args = {'fontsize': 9, 'alpha': 0.8}
                if font_prop:
                     text_args['fontproperties'] = font_prop

                # 直接绘制文本标签 (不添加到 texts 列表，因为不使用 adjustText)
                if idx < X.shape[0]: # 避免索引超出范围
                     ax.text(X[idx, 0], X[idx, 1], label_text, **text_args)
                else:
                     print(f"Warning: Index {idx} out of bounds for X ({X.shape[0]} rows) during text label generation. Skipping text label.")

    # --- 提示文本标签是否已绘制 ---
    if plot_labels:
        print("Plotting text labels beside points.")
        print("Note: adjustText is not used, labels may overlap.")
    else:
        print("Plotting text labels is disabled (plot_labels=False).")


    # --- 设置标题、轴标签和网格 (使用 font_prop) ---
    title_props = {'fontsize': 16, 'fontweight': 'bold'}
    label_props = {'fontsize': 12}
    if font_prop:
        title_props['fontproperties'] = font_prop
        label_props['fontproperties'] = font_prop

    ax.set_title(plot_title, **title_props)
    ax.set_xlabel('Component 1', **label_props)
    ax.set_ylabel('Component 2', **label_props)
    ax.grid(True, linestyle='--', alpha=0.5)


    # --- 创建图例 (使用 font_prop) ---
    legend_labels = [str(name) for name in unique_labels] # 确保图例标签是字符串
    ncol = 1 if num_unique_labels <= 20 else 2
    legend_kwargs = {'loc': 'upper left', 'bbox_to_anchor': (1.03, 1.0),
                     'title': label_title, 'ncol': ncol,
                     'fontsize': 10, 'markerscale': 1.0}
    if font_prop:
        legend_kwargs['prop'] = font_prop
        # 设置图例标题字体（兼容不同 matplotlib 版本）
        # legend_title_font = font_prop.copy()
        # legend_title_font.set_size(12)
        # legend_kwargs['title_fontproperties'] = legend_title_font # 需要 matplotlib 3.3+

    # 仅当有句柄时才创建图例
    if handles:
        legend = ax.legend(handles=handles, labels=legend_labels, **legend_kwargs)
        # 兼容旧版本 matplotlib 设置图例标题字体的方式
        if font_prop and 'title_fontproperties' not in legend_kwargs:
             try:
                 plt.setp(legend.get_title(), fontproperties=font_prop)
             except Exception as e:
                 #print(f"Warning: Could not set legend title font using fallback method: {e}")
                 pass # 避免不必要的警告

        # --- 调整布局以适应图例 ---
        plt.subplots_adjust(right=0.8 if ncol == 1 else 0.7)

    else:
        print("Warning: No data points plotted, skipping legend creation.")


    # --- 保存图像 ---
    try:
         plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
         print(f"Saved plot to {save_path}")
    except Exception as e:
         print(f"Error saving plot {save_path}: {e}")
    finally:
        plt.close() # 确保图形关闭，释放内存