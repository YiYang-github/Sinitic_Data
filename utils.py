import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 导入 font_manager

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