import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 用户配置区域 (修改这里) =================

# 数据文件夹路径
base_dir = r"D:\数据资料\联影电影数据\rarvproj"

# 【重点】在此处填入你想剔除的影像号 (PatientID)
# 格式：用方括号括起来，中间用逗号分隔。可以是数字或字符串。
# 例如：EXCLUDE_IDS = [12414937, 12466637]
EXCLUDE_IDS = [
    12414937,
    12505899,  # 比如这个是你觉得LV异常大的
    #12487255,  # 比如这个可能是分割有误的
    # 你可以继续在这里添加...
]

# 图片保存文件夹名称
output_folder_name = "Result_Plots_Manual_Removal"

# ================= 2. 数据读取与处理 =================

output_dir = os.path.join(base_dir, output_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘图风格设置
sns.set_theme(style="whitegrid")
palette = {"Men": "#E37C5B", "Women": "#74B796"}

print("正在读取数据...")

try:
    # 读取文件
    df_basic = pd.read_csv(os.path.join(base_dir, "基本信息1.csv"))
    df_lh = pd.read_csv(os.path.join(base_dir, "left_heart_metrics_indexed_v2.csv"))
    df_rh = pd.read_csv(os.path.join(base_dir, "right_heart_metrics_indexed_v2.csv"))
except Exception as e:
    print(f"错误：读取文件失败。请检查路径。具体错误：{e}")
    exit()

# 统一性别表达
df_basic['Sex_En'] = df_basic['性别'].map({'男': 'Men', '女': 'Women'})

# 确定连接键 (兼容 '影像号' 或 '编号')
join_key = '影像号' if '影像号' in df_basic.columns else '编号'

# 转换 ID 格式为字符串，确保匹配不出错
df_basic[join_key] = df_basic[join_key].astype(str)
df_lh['PatientID'] = df_lh['PatientID'].astype(str)
df_rh['PatientID'] = df_rh['PatientID'].astype(str)
exclude_ids_str = [str(x) for x in EXCLUDE_IDS] # 将用户输入的ID也转为字符串

# 合并数据
df_merged = pd.merge(df_basic, df_lh, left_on=join_key, right_on='PatientID')
df_merged = pd.merge(df_merged, df_rh, on='PatientID')

# 重命名年龄
if '年龄' in df_merged.columns:
    df_merged.rename(columns={'年龄': 'Age'}, inplace=True)

# 记录原始数量
original_count = len(df_merged)

# ================= 3. 执行剔除逻辑 =================

# 过滤数据
df_final = df_merged[~df_merged['PatientID'].isin(exclude_ids_str)]

# 计算被剔除的数量
removed_count = original_count - len(df_final)

print("-" * 30)
print(f"原始样本数: {original_count}")
print(f"手动剔除 ID: {EXCLUDE_IDS}")
print(f"剔除数量: {removed_count}")
print(f"最终绘图样本数: {len(df_final)}")
print("-" * 30)

# ================= 4. 定义绘图指标 (含LA/RA) =================

metrics_map = {
    # --- 左室 ---
    'LVEDVi': 'LV EDVi ($mL/m^2$)',
    'LVESVi': 'LV ESVi ($mL/m^2$)',
    'LVSVi':  'LV SVi ($mL/m^2$)',
    'LVEF_%': 'LV EF (%)',
    # --- 右室 ---
    'RVEDVi': 'RV EDVi ($mL/m^2$)',
    'RVESVi': 'RV ESVi ($mL/m^2$)',
    'RVSVi':  'RV SVi ($mL/m^2$)',
    'RVEF_%': 'RV EF (%)',
    # --- 左房 ---
    'LAVmaxi': 'LA Max Vol Index ($mL/m^2$)',
    'LAVmini': 'LA Min Vol Index ($mL/m^2$)',
    'LA_reservoir_EF_%': 'LA EF (%)',
    # --- 右房 ---
    'RAVmaxi': 'RA Max Vol Index ($mL/m^2$)',
    'RAVmini': 'RA Min Vol Index ($mL/m^2$)',
    'RA_reservoir_EF_%': 'RA EF (%)'
}

# ================= 5. 循环绘图 =================

for col, label in metrics_map.items():
    plt.figure(figsize=(8, 6))
    
    # 绘制男性
    sns.regplot(
        data=df_final[df_final['Sex_En']=='Men'], 
        x='Age', y=col, 
        color=palette['Men'], 
        scatter_kws={'alpha':0.6, 's':50}, 
        label='Men'
    )
    # 绘制女性
    sns.regplot(
        data=df_final[df_final['Sex_En']=='Women'], 
        x='Age', y=col, 
        color=palette['Women'], 
        scatter_kws={'alpha':0.6, 's':50},
        label='Women'
    )
    
    plt.xlabel('Age (years)', fontsize=12, fontweight='bold')
    plt.ylabel(label, fontsize=12, fontweight='bold')
    plt.title(f'{col} vs Age (n={len(df_final)})', fontsize=14)
    plt.legend(title='Sex')
    
    # 优化坐标轴范围
    y_min, y_max = df_final[col].min(), df_final[col].max()
    margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 5
    plt.ylim(max(0, y_min - margin), y_max + margin)
    plt.xlim(df_final['Age'].min() - 5, df_final['Age'].max() + 5)
    
    plt.tight_layout()
    
    # 保存图片
    save_name = f"{col}_ManualSelect.png".replace("%", "pct")
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_name}")

print(f"\n全部完成！图片保存在: {output_dir}")