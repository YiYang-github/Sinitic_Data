<<<<<<< HEAD
# Overview

In this project, we release the most diverse and comprehensive features for hundreds of Sinitic dialects from large-scale transcription and raw waves to faciliate further research in language variation/classification/evolution or related domains like Economics as references.

## 0. Data Used

We collected raw speech ('Data2'), transcription ('Data4'), categorical annotation ('Data3') and historical information('Data1'). Below is detailed introduction.

**表 1: 数据集概览**
| 标识  | 内容  | 规模  |  特征  | 来源 |
|--------------|--------------------------------------|-------------------------------------------------------------------------|---------------------------------------------|----------|
| `Data 1` | 中古音韵 | 3804字，表格  |          | closed source |
| `Data 2` | 汉语方言音频 | 827GB音频  |  原始语音特征  | closed source |
| `Data 3` | 《汉语方言地图集》 | 表格  |  语音、词汇、语法特征  | [^huang2024][^yang2024] |
| `Data 4` | 中国语言资源保护工程  |  1289地1000词的表格  |  声母、韵母、声调特征  | [^1] |

[^1]: <https://zhongguoyuyan.cn/index>
[^huang2024]: Huang, H., Grieve, J., Jiao, L., & Cai, Z. (2024). Geographic structure of Chinese dialects: a computational dialectometric approach. Linguistics, 62(4), 937-976.
[^yang2024]: Yang, C., Zhang, X., Yan, S., Yang, S., Wu, B., You, F., ... & Zhang, M. (2024). Large-scale lexical and genetic alignment supports a hybrid model of Han Chinese demic and cultural diffusions. Nature Human Behaviour, 8(6), 1163-1176.
[^zhang2019]: Zhang, M., Yan, S., Pan, W., & Jin, L. (2019). Phylogenetic evidence for Sino-Tibetan origin in northern China in the Late Neolithic. Nature, 569(7754), 112-115.

## 1. Quick Use

## Data2: Speech Representations 

## Data3: Phonology, Lexicon, Syntax Representations

## Data4: Initial, Final, and Tone Representations

transcription_areas.pkl: dict
[initials, final, tone] + [area, slice, slices] + word_name + coords


