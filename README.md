# Overview

In this project, we release the most diverse and comprehensive features for hundreds of Sinitic dialects from large-scale transcription and raw waves to faciliate further research in language variation/classification/evolution or related domains like Economics as references.

You can use the [load.py](load.py) to load representations directly.

## 0. Data Used

We collected raw speech ('Data2'), transcription ('Data4'), categorical annotation ('Data3') and historical information('Data1'). For each dataset, we apply clear and consistent preprocessing. Below is detailed introduction.

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


## 1. Automatic Labeling of Dialect Area & Slice 

<div align="center">
<img align="center" src="imgs/Auto_dialect.png" width="800px" />
<b><br>Fig 3. Dialect Locations for Data3</b>
</div>

To ensure consistent dialect classification, we developed an AI agent that automatically labels Sinitic dialect areas (方言大区) and slices (方言片) for given geographical inputs. This addresses inconsistencies across datasets and automates the laborious manual lookup process.

**Core Workflow:**
1.  **Input**: User provides latitude/longitude or textual location descriptions.
2.  **Geocoding**: Textual descriptions are converted to coordinates and standardized administrative information (province, city, county) using a geocoding service (e.g., Tencent Maps API).
3.  **Knowledge Base Matching**: The county-level information is matched against our **Dialectal Geospatial Knowledge Base**. This knowledge base was constructed by digitizing and structuring 23 dialect maps from the *Language Atlas of China* (2nd Edition) using Vision-Language Models (VLMs like Gemini 2.5 Pro) for map parsing, Large Language Models (LLMs) for refinement, and manual verification. It maps counties to their respective dialect areas/slices.
4.  **Output**: The agent returns the corresponding dialect area and slice.



## Data1: Middle Chinese Phonological Information 

`Data1` offers a structured representation of Middle Chinese (中古汉语) phonological information for a list of Chinese characters, essential for historical phonology and dialect evolution studies. The processed data is available in `Data1/data.pkl`.

**Content:**
The dataset provides numerically encoded features for 3201 cleaned characters:
* `word`: List of Chinese characters.
* `initial`: Numerical codes for Middle Chinese initials (古声母).
* `final_1`: Numerical codes for the first structural part of the Middle Chinese rhyme (e.g., *Shè* 摄).
* `final_2`: Numerical codes for the second structural part (e.g., *Hū* 呼 - open/closed mouth & *Děng* 等 - division).
* `final_3`: Numerical codes for the specific rhyme name (韵目).
* `tone`: Numerical codes for Middle Chinese tones (古声调).

**Loading Example:**
```python
data1_info = load_feats(name='Data1', type='info')

# Access components:
character_list = data1_info['word']
initial_codes = data1_info['initial']
# ... and so on
```

Running this command will typically show loading progress or information similar to this output:

```text
正在从文件 'Data1/data.pkl' 加载数据...
计划加载的特征: ['word', 'initial', 'final_1', 'final_2', 'final_3', 'tone']
成功加载 6 个特征。
```

**Processing Highlights:**
The raw data (3807 entries) was processed by:
1.  Extracting single characters.
2.  Decomposing the complex "韵部" (rhyme part) string into three structural components.
3.  Mapping phonological categories (initials, tones, and the three rhyme components) to numerical IDs.
4.  Cleaning by removing duplicates and entries with parsing issues.
The unique numerical categories for the three processed rhyme components are 16, 8, and 202, respectively. The original dataset had 44 unique initials, 284 unique raw rhyme strings, and 4 tones.



## Data2: Speech-based Representations 

This section details speech-based representations extracted from `Data2` (827GB of Sinitic dialect audio). These features aim to capture acoustic properties of different dialectal regions and slices.

### 2.1 MFCCs-Based Features

The foundational acoustic features are 39-dimensional Mel-Frequency Cepstral Coefficients (MFCCs), extracted every 10 ms from the raw audio waveforms. Based on these MFCCs, we derive two types of higher-level representations:

**1. Mean MFCC Vectors**

For each dialect area and dialect slice, a single, representative 39-dimensional feature vector is computed by averaging all MFCC frames belonging to that specific area or slice. This provides a general acoustic profile for each dialectal grouping.

You can load these mean-pooled representations using:

```python
# Load mean MFCCs aggregated by dialect area
dialect_mean_mfccs = load_feats(name='Data2', type='mfcc_dialect_mean') 
# Expected content: {'features': np.array (17, 39), 'names': np.array (17,)}

# Load mean MFCCs aggregated by dialect slice
slice_mean_mfccs = load_feats(name='Data2', type='mfcc_slice_mean')
# Expected content: {'features': np.array (77, 39), 'names': np.array (77,)}
```

* The `dialect_mean_mfccs` dictionary contains:
    * `features`: A NumPy array of shape `(17, 39)` representing the mean MFCC vector for each of the 17 dialect areas.
    * `names` (or `dialect_names`): A NumPy array of shape `(17,)` containing the names of these dialect areas.
* The `slice_mean_mfccs` dictionary contains:
    * `features`: A NumPy array of shape `(77, 39)` representing the mean MFCC vector for each of the 77 dialect slices.
    * `names` (or `slice_names`): A NumPy array of shape `(77,)` containing the names of these dialect slices.

**2. GMM-i-vector Representations**

To capture more complex distributional characteristics of the MFCCs for each dialect area, we also trained a Gaussian Mixture Model (GMM) based i-vector system. A Universal Background Model (UBM) with 256 Gaussian components was first trained on a large subset of the MFCC data. Subsequently, a 400-dimensional i-vector was extracted for each dialect area, representing its acoustic characteristics within the total variability space.

You can load these i-vector representations using:

```python
dialect_gmm_ivectors = load_feats(name='Data2', type='mfcc_dialect_gmm_ivector')
# Expected content: {'features': np.array (17, 400), 'names': np.array (17,)}
```

* The `dialect_gmm_ivectors` dictionary contains:
    * `features`: A NumPy array of shape `(17, 400)` representing the 400-dimensional i-vector for each of the 17 dialect areas.
    * `names` (or `dialect_names`): A NumPy array of shape `(17,)` containing the names of these dialect areas.

### 2.2 Pretrained Speech Models (基于预训练模型的特征)

Representations derived from state-of-the-art pre-trained speech models (e.g., Wav2Vec2.0, HuBERT, Whisper, WavLM) will be extracted and made available in a future update. These models, pre-trained on vast amounts of speech data, are expected to provide highly robust and generalizable features, often less sensitive to variations in recording equipment and noise compared to traditional MFCCs.

*(To be released soon)*




## Data3: Phonology, Lexicon, Syntax Representations

<div align="center">
<img align="center" src="imgs/LACD.png" width="500px" />
<b><br>Fig 3. Dialect Locations for Data3</b>
</div>

As seen in Fig 3 (left), the raw dataset contains 930 Sinitic dialects. For each dialect, linguists documentated 205 phonology、203 lexicon and 102 syntax maps.

You can load the distance matrix and related coordinates using the `load_feats` function in `load.py`. Specifically, use `type='distance_matrices'` and `type='info'`  to get this initial set of data:

```python
distance_matricas = load_feats(name='Data4', type='distance_matrices')
info = load_feats(name='Data3', type='info')
```

Running this command will typically show loading progress or information similar to this output:

```text
正在从文件 'Data3/distance_matrices.npz' 加载数据...
计划加载的特征: ['lexicon_distance', 'phonology_distance', 'syntax_distance', 'overall_distance']
成功加载 4 个特征。

正在从文件 'Data3/info.npz' 加载数据...
计划加载的特征: ['coords']
成功加载 1 个特征。
```

## Data4: Initial, Final, and Tone Representations

<div align="center">
<img align="center" src="imgs/Data4.png" width="1000px" />
<b><br>Fig 4: (Left) Original 1289 Dialect Locations for Data4; (Right) 1084 Dialect Locations for Processed Data4</b>
</div>


### 4.1: Raw Transcription Loading

As seen in Fig 4 (left), the raw dataset contains 1289 Sinitic dialects. For each dialect, linguists investigate 1000 words.

You can load the original transcription data and related metadata using the `load_feats` function in `load.py`. Specifically, use `type='raw'` to get this initial set of data:

```python
raw_data_dict = load_feats(name='Data4', type='raw')
```

Running this command will typically show loading progress or information similar to this output:

```text
正在从文件 'Data4/transcription_areas.pkl' 加载数据...
计划加载的特征: ['word_name', 'area', 'slice', 'slices', 'coords', 'initial', 'final', 'tone']
成功加载 8 个特征。
```

The `raw_data_dict` dictionary will contain the following keys, corresponding to the loaded features:

* **Original Transcriptions:**
    * `initial`: Original transcription of initials. (`numpy` array, shape `[1289, 999]`, string type)
    * `final`: Original transcription of finals. (`numpy` array, shape `[1289, 999]`, string type)
    * `tone`: Original transcription of tones. (`numpy` array, shape `[1289, 999]`, string type)
    * *Note:* The second dimension is 999 instead of 1000 because the word '0053 瓦' is fully missing across all 1289 dialects in the original data source for these transcription features.

* **Locations and Classifications:**
    * `word_name`: A list or array of the 1000 words being investigated (including a placeholder or indicator for '0053 瓦').
    * `area`: Classification of dialect areas.
    * `slice`, `slices`: Further classifications or groupings of these areas.
    * `coords`: Geographic coordinates for each dialect.

You can then access the individual data components using the dictionary keys, for example:

```python
initials_data = raw_data_dict['initial']
dialect_locations = raw_data_dict['coords']
```


### 4.2 Data Processing and Distance Matrix Calculation


Before calculating dialect distances, the raw transcription data undergoes a processing step to handle missing and potentially unreliable values.

The criteria for setting a transcription as a missing value were:
* The transcription was marked as 'wrong' in the original data.
* The specific transcription (of an initial, final, or tone) appeared less than 1000 times across the entire dataset, indicating low frequency and potential unreliability.

After marking these values as missing, we filtered both the dialects (rows) and the words (columns). We kept only those dialects and words where the proportion of **missing values was less than 30%**.

Following this filtering operation, the dataset was reduced to **1084 dialects** (rows) and **915 words** (columns).

Subsequently, dialect distance matrices were calculated pairwisely based on the remaining **915 features** (words). Separate distance matrices were computed for:
* Initials (`initials_distance`)
* Finals (`finals_distance`)
* Tones (`tones_distance`)
* An overall combined distance (`overall_distance`)

Each resulting distance matrix has a shape of `[1084, 1084]`.

You can load these distance matrices using the `load_feats` function with `type='distance_matrices'`:

```python
distance_matrices_dict = load_feats(name='Data4', type='distance_matrices')

# Example: Access the initials distance matrix
initial_distance_matrix = distance_matrices_dict['initials_distance']

# You can similarly access 'finals_distance', 'tones_distance', and 'overall_distance'
# final_distance_matrix = distance_matrices_dict['finals_distance']
# etc.
```

Simultaneously, you can load the associated metadata (such as coordinates, areas, and classifications) for the **filtered dialects** using `type='info'`:

```python
info_dict = load_feats(name='Data4', type='info')

# Access the associated metadata for the 1084 kept dialects
coords_data = info_dict['coords']
areas_data = info_dict['areas']
slice_data = info_dict['slice']
slices_data = info_dict['slices']
word_names_data = info_dict['word_names'] # Note: word_names is also included in 'info'
```