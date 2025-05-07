from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def read_pdb_to_dataframe(pdb_path: Optional[str] = None, model_index: int = 1, parse_header: bool = True) -> pd.DataFrame:
    atomic_df = PandasPdb().read_pdb(pdb_path)
    if parse_header:
        header = parsePDBHeader(pdb_path)
    else:
        header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")
    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header

df, df_header = read_pdb_to_dataframe('pdb1j9l.ent')
df.head(10)

def calculate_distance(row, ligand_coords):
    return np.sqrt(
        (row['x_coord'] - ligand_coords[0]) ** 2 +
        (row['y_coord'] - ligand_coords[1]) ** 2 +
        (row['z_coord'] - ligand_coords[2]) ** 2
    )


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

drive_path = '/content/drive/MyDrive/Model Training Data/Mn/Only First Ligand'
print(drive_path)
print(os.listdir(drive_path))

file_path = os.path.join(drive_path, 'Mn_First Ligand Only_Amino Acid Counts.csv')
print(file_path)

##main algorithm for proccesing the DATA
if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    dataset_path = '/content/drive/MyDrive/Model Training Data/Mg/Only First Ligand/'
    searched_ligand = 'Mg'
    amino_acids_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    all_amino_acid_counts = {aa: 0 for aa in amino_acids_list}
    all_angles = []
    log_file_path = '/content/drive/MyDrive/logs/processing_log.txt'

    with open(log_file_path, "w") as log_file:
        processed_files_count = 0
        failed_files = []

        for filename in os.listdir(dataset_path):
            if filename.endswith((".ent", ".pdb")):
                file_path = os.path.join(dataset_path, filename)
                log_file.write(f"Processing file: {filename}\n")
                try:
                    df, df_header = read_pdb_to_dataframe(file_path)
                except Exception as e:
                    log_file.write(f"Error processing file {filename}: {e}\n")
                    failed_files.append(filename)
                    continue

                ligand_location = df.loc[df['atom_name'] == searched_ligand, ['x_coord', 'y_coord', 'z_coord']]
                if not ligand_location.empty:
                    first_ligand_location = ligand_location.head(1)
                    first_ligand_coords = first_ligand_location.values[0]

                    df['distance_to_ligand'] = df.apply(lambda atom_record: calculate_distance(atom_record, first_ligand_coords), axis=1)

                    ligands_nearby_atoms = df[df['distance_to_ligand'] <= 5]
                    ligands_nearby_atoms = ligands_nearby_atoms[ligands_nearby_atoms['atom_name'] != searched_ligand]

                    result = ligands_nearby_atoms[['record_name', 'atom_name', 'residue_name', 'chain_id', 'residue_number', 'x_coord', 'y_coord', 'z_coord', 'distance_to_ligand']]

                    # Filter out ATOM type atoms for counting amino acids in the binding site
                    result = result[result['record_name'] == 'ATOM']

                    unique_residues = result.drop_duplicates(subset=['residue_name', 'residue_number'])
                    amino_acid_counts = unique_residues.groupby('residue_name').size()

                    for aa, count in amino_acid_counts.items():
                        if aa in amino_acids_list:
                            all_amino_acid_counts[aa] += count

                    hetatm_vo4_nearby = ligands_nearby_atoms[
                        (ligands_nearby_atoms['record_name'] == 'HETATM') &
                        (ligands_nearby_atoms['residue_name'] == 'VO4')
                    ]
                    if not hetatm_vo4_nearby.empty:
                        coordinates_hetatm_vo4_nearby = hetatm_vo4_nearby[['x_coord', 'y_coord', 'z_coord']].values
                        num_atoms = len(coordinates_hetatm_vo4_nearby)

                        if num_atoms > 1:
                            for i in range(num_atoms):
                                for j in range(i + 1, num_atoms):
                                    angle = calculate_angle(
                                        coordinates_hetatm_vo4_nearby[i],
                                        first_ligand_coords,
                                        coordinates_hetatm_vo4_nearby[j]
                                    )
                                    if angle is not None:
                                        all_angles.append(angle)

                processed_files_count += 1
                log_file.write(f"File {filename} processed successfully.\n")

        log_file.write(f"\nTotal files processed: {processed_files_count}\n")
        log_file.write(f"Failed files: {failed_files}\n")

    print("\nTotal Amino Acid Counts:")
    print(all_amino_acid_counts)
    print("\nAll Angles:")
    print(all_angles)
    print(f"Log file saved to: {log_file_path}")

def process_ligand_data(dataset_path, searched_ligand):
    amino_acid_counts_df = pd.read_csv(os.path.join(dataset_path, f'{searched_ligand} - First Ligand Only - Amino Acid Counts.csv'))
    amino_acid_counts_df = amino_acid_counts_df.iloc[:-1]  

    distances_df = pd.read_csv(os.path.join(dataset_path, f'{searched_ligand} - First Ligand Only - Distances.csv'))
    angles_df = pd.read_csv(os.path.join(dataset_path, f'{searched_ligand} - First Ligand Only - Angles.csv'))

    return amino_acid_counts_df, distances_df, angles_df

amino_acid_counts_df, distances_df, angles_df = process_ligand_data(dataset_path, searched_ligand)
print(amino_acid_counts_df.head())
print(distances_df.head())

print(angles_df.head())

amino_acid_counts = amino_acid_counts_df.drop('PDB ID', axis=1).sum()
amino_acid_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Amino Acid Distribution in Binding Sites')
plt.xlabel('Amino Acid')
plt.ylabel('Count')
plt.show()


angles_data = pd.melt(angles_df, id_vars='pdb_file_name', value_vars=[col for col in angles_df.columns if col != 'pdb_file_name'], value_name='angle')

plt.figure(figsize=(10, 6))
sns.histplot(angles_data['angle'], bins=30, kde=True)
plt.title('Angle Distribution')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=angles_data['angle'])
plt.title('Angle Distribution (Boxplot)')
plt.ylabel('Angle (degrees)')
plt.show()

print(angles_data['angle'].describe())

print("Skewness:", angles_data['angle'].skew())
print("Kurtosis:", angles_data['angle'].kurtosis())

print(angles_data.head(5))
print("Columns in distances_df:", distances_df.columns)

distance_columns = [col for col in distances_df.columns if col.startswith('distance')]

first_row = distances_df.iloc[0]

print("Distance values in the first row:")
print(first_row[distance_columns])

distances = first_row[distance_columns].dropna().values
print("\nNon-NaN distance values in the first row:")
print(distances)

if distances.size > 0:
    mean_val = np.mean(distances)
    median_val = np.median(distances)
    std_val = np.std(distances)
    min_val = np.min(distances)
    max_val = np.max(distances)
    print("\nCalculated statistics for the first row:")
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Std: {std_val}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
else:
    print("\nNo valid distance values found in the first row.")

# חשב סטטיסטיקות על פני עמודות המרחק עבור כל pdb_file_name
def calculate_row_stats(row):
    distances = row[distance_columns].dropna().values
    if distances.size > 0:
        return pd.Series({
            'distance_mean': np.mean(distances),
            'distance_median': np.median(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances)
        })
    else:
        return pd.Series({
            'distance_mean': np.nan,
            'distance_median': np.nan,
            'distance_std': np.nan,
            'distance_min': np.nan,
            'distance_max': np.nan
        })

df_distances_summary = distances_df.groupby('pdb_file_name').apply(calculate_row_stats).reset_index()

print("\ndf_distances_summary head:")
print(df_distances_summary.head())

distance_columns = [col for col in distances_df.columns if col.startswith('distance')]


results = []
for name, group in distances_df.groupby('pdb_file_name'):
    distances = group[distance_columns].dropna(axis=1, how='all').values.flatten()
    distances = distances[~np.isnan(distances)] # הסרת NaN-ים מהמערך השטוח
    if distances.size > 0:
        results.append({
            'pdb_file_name': name,
            'distance_mean': np.mean(distances),
            'distance_median': np.median(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances)
        })
    else:
        results.append({
            'pdb_file_name': name,
            'distance_mean': np.nan,
            'distance_median': np.nan,
            'distance_std': np.nan,
            'distance_min': np.nan,
            'distance_max': np.nan
        })

df_distances_summary = pd.DataFrame(results)

print(df_distances_summary.head())

angle_columns = [col for col in angles_df.columns if col.startswith('angle')]

angles_mean_median = angles_df.groupby('pdb_file_name')[angle_columns].apply(lambda x: pd.Series({
    'angle_mean': x.mean().mean(),
    'angle_median': pd.Series(x.values.flatten()).median()

}))
print(angles_mean_median.head())

angle_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
angle_labels = [f'{angle_bins[i]}-{angle_bins[i+1]}' for i in range(len(angle_bins) - 1)]

df_angles_ranges = angles_df.groupby('pdb_file_name')[angle_columns].apply(
    lambda x: pd.cut(x.stack(), bins=angle_bins, labels=angle_labels, right=False).value_counts()
).unstack(fill_value=0).add_prefix('angle_range_')


df_angles_final = pd.concat([df_angles_ranges, angles_mean_median], axis=1)
print(df_angles_final.head())


# mergging df_distances_summary with df_angles_final
merged_df = pd.merge(df_distances_summary, df_angles_final, on='pdb_file_name', how='outer')
amino_acid_counts_df = amino_acid_counts_df.rename(columns={'PDB ID': 'pdb_file_name'})
final_merged_df = pd.merge(merged_df, amino_acid_counts_df, on='pdb_file_name', how='outer')
final_merged_df['label'] = 'Mg'
print(final_merged_df.head(50))

output_path = '/content/drive/MyDrive/final_merged_df.csv'  # שנה את הנתיב ושם הקובץ לפי הצורך
final_merged_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"הקובץ נשמר בהצלחה בנתיב: {output_path}")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
dataset_path = '/content/drive/MyDrive/PDB/'  
filename_mn = 'df_final_merged_mn.csv'
filename_mg = 'df_final_merged_mg.csv'

try:
    df_mn = pd.read_csv(os.path.join(dataset_path, filename_mn))
    df_mg = pd.read_csv(os.path.join(dataset_path, filename_mg))
    print("loaded.")
except FileNotFoundError:
    print("Please check path")
    df_mn = None
    df_mg = None

if df_mn is not None and df_mg is not None:
    df_combined = pd.concat([df_mn, df_mg], ignore_index=True)
    print("\n DataFrames conncat.")
    print(df_combined.head())
    print(f"\n DataFrames conncat.: {len(df_combined)}")

    output_filename_combined = 'df_final_combined_mn_mg.csv'
    output_path_combined = os.path.join(dataset_path, output_filename_combined)

    df_combined.to_csv(output_path_combined, index=False, encoding='utf-8')
    print(f"\n save: {output_path_combined}")

dataset_path = '/content/drive/MyDrive/PDB/'
file_combined = 'df_final_combined_mn_mg.csv'
try:
    df_final = pd.read_csv(os.path.join(dataset_path, file_combined))
    df_final = df_final.drop('pdb_file_name', axis=1)

    # --- Check for missing values ---
    missing_values = df_final.isnull().sum()
    print("\nNumber of missing values per column before imputation:")
    print(missing_values)
    # --- End of checking for missing values ---

    print("Success.")
except FileNotFoundError:
    print("Fail.")
    df_final = None


# Creating Baseline

# ---  MajorityClassClassifier ---
class MajorityClassClassifier:
    def fit(self, X, y):
        self.majority_class = Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.full(X.shape[0], self.majority_class)

# --- RandomClassifier ---
class RandomClassifier:
    def fit(self, X, y):
        class_counts = Counter(y)
        total_samples = len(y)
        self.class_probabilities = {cls: count / total_samples for cls, count in class_counts.items()}
        self.classes = list(self.class_probabilities.keys())
        self.probabilities = list(self.class_probabilities.values())

    def predict(self, X):
        return np.random.choice(self.classes, size=X.shape[0], p=self.probabilities)

# ---  evaluate_baseline ---
def evaluate_baseline(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_roc_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

        if len(np.unique(y)) > 1:
            try:
                auc_roc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1], multi_class='ovr') if hasattr(model, 'predict_proba') else np.nan
                auc_roc_scores.append(auc_roc)
            except ValueError:
                auc_roc_scores.append(np.nan)
        else:
            auc_roc_scores.append(np.nan)

    print(f"Results for {model.__class__.__name__}:")
    print(f"  Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"  Average Precision: {np.mean(precision_scores):.4f}")
    print(f"  Average Recall: {np.mean(recall_scores):.4f}")
    print(f"  Average F1-score: {np.mean(f1_scores):.4f}")
    print(f"  Average AUC-ROC: {np.mean(auc_roc_scores):.4f}")
    print("-" * 30)

dataset_path = '/content/drive/MyDrive/PDB/'
file_combined = 'df_final_combined_mg_mn.csv'
# df_final = pd.read_csv(os.path.join(dataset_path, file_combined), index_col=0)
numerical_cols = df_final.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
df_final[numerical_cols] = imputer.fit_transform(df_final[numerical_cols])
X = df_final.drop('label', axis=1)
y_labels = df_final['label']
y = np.where(y_labels == 'Mg', 0, 1)

print("\n--- הערכת מודלים בסיסיים (Baselines) על כל הדאטה ---")
majority_clf = MajorityClassClassifier()
random_clf = RandomClassifier()

evaluate_baseline(X.values, y, majority_clf)
evaluate_baseline(X.values, y, random_clf)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])


# ---  Histo and Plots ---
import matplotlib.pyplot as plt
import numpy as np

baseline_majority = {'Accuracy': 0.8257, 'Precision': 0.6817, 'Recall': 0.8257, 'F1-score': 0.7468}
baseline_random = {'Accuracy': 0.7076, 'Precision': 0.7080, 'Recall': 0.7076, 'F1-score': 0.7078}
random_forest = {'Accuracy': 0.7655, 'Precision': 0.9228, 'Recall': 0.7815, 'F1-score': 0.8462}
svm = {'Accuracy': 0.7898, 'Precision': 0.9163, 'Recall': 0.8205, 'F1-score': 0.8656}

models = ['Majority Class', 'Random', 'Random Forest', 'SVM']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

values = {
    'Majority Class': [baseline_majority[metric] for metric in metrics],
    'Random': [baseline_random[metric] for metric in metrics],
    'Random Forest': [random_forest[metric] for metric in metrics],
    'SVM': [svm[metric] for metric in metrics]
}

x = np.arange(len(metrics)) 
width = 0.2 

fig, ax = plt.subplots(figsize=(10, 6))
rects = []
for i, model in enumerate(models):
    offset = (i - len(models) // 2) * width + (width / 2 if len(models) % 2 == 0 else 0)
    rect = ax.bar(x + offset, values[model], width, label=model)
    rects.append(rect)


ax.set_ylabel('Score')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

ax.bar_label(rects[0], padding=3, fmt='%.3f')
ax.bar_label(rects[1], padding=3, fmt='%.3f')
ax.bar_label(rects[2], padding=3, fmt='%.3f')
ax.bar_label(rects[3], padding=3, fmt='%.3f')

fig.tight_layout()
plt.show()

# ---end Histo and Plots---

if df_final is not None:
    #Pairplo
    num_cols_for_pairplot = min(5, len(df_final.columns))
    cols_for_pairplot = df_final.iloc[:, :num_cols_for_pairplot].copy()
    cols_for_pairplot['label'] = df_final['label'] 
    sns.pairplot(cols_for_pairplot, hue='label')
    plt.title('Pairplot (Part of features)')
    plt.show()

# Separate features (X) and labels (y)
X = df_final.drop('label', axis=1)
y = df_final['label']
positive_label = 'Mg'
class_names = np.unique(y) # Get unique class names for printing

# Identify numerical columns
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# --- Handle missing values ---
imputer = SimpleImputer(strategy='median')
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
print("\nMissing values in numerical columns imputed using the median.")
# --- End of handling missing values ---


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling using StandardScaler on numerical columns only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("\nScaled Training Set (first 5 rows):\n", X_train_scaled[:5])
print("\nScaled Testing Set (first 5 rows):\n", X_test_scaled[:5])

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42) # probability=True for ROC AUC
    }

results = {}

for model_name, model in models.items():
    print(f"\nPerforming Cross-Validation for {model_name}:")
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_scaled, y_train)):
        X_train_fold = X_train_scaled.iloc[train_index].copy()
        X_val_fold = X_train_scaled.iloc[val_index].copy()
        y_train_fold = y_train.iloc[train_index].copy()
        y_val_fold = y_train.iloc[val_index].copy()

        # --- Feature Selection using Mutual Information ---
        selector = SelectKBest(mutual_info_classif, k=10) # בחר את 10 הפיצ'רים הטובים ביותר
        selector.fit(X_train_fold, y_train_fold)

        # הדפסת ה-MI של הפיצ'רים
        mi_scores = pd.Series(selector.scores_, index=X_train_fold.columns)
        print(f"\nFold {fold + 1} - Mutual Information Scores:")
        print(mi_scores.sort_values(ascending=False).head(10))

        # קבלת אינדקסים ושמות הפיצ'רים הנבחרים
        selected_features_indices = selector.get_support(indices=True)
        selected_features = X_train_fold.columns[selected_features_indices]
        print(f"\nFold {fold + 1} - Selected Features: {selected_features}")

        # החלת בחירת הפיצ'רים על פולדי האימון והוולידציה
        X_train_fold_selected = X_train_fold[selected_features]
        X_val_fold_selected = X_val_fold[selected_features]
        # --- End of Feature Selection ---

        # Train the model
        model.fit(X_train_fold_selected, y_train_fold) # אימון על הפיצ'רים הנבחרים

        # Make predictions on the validation set
        y_pred_fold = model.predict(X_val_fold_selected) # תחזית על הפיצ'רים הנבחרים
        y_pred_proba_fold = model.predict_proba(X_val_fold_selected)[:, 1] # הסתברויות על הפיצ'רים הנבחרים

        # Evaluate the model
        accuracy = accuracy_score(y_val_fold, y_pred_fold)
        precision = precision_score(y_val_fold, y_pred_fold, average='binary', pos_label='Mg')
        recall = recall_score(y_val_fold, y_pred_fold, average='binary', pos_label='Mg')
        f1 = f1_score(y_val_fold, y_pred_fold, average='binary', pos_label='Mg')
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba_fold)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)

        print(f"  Fold {fold + 1}:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-score: {f1:.4f}")
        print(f"    AUC: {roc_auc:.4f}")

    # Store the average results for the current model
    results[model_name] = {
        'Accuracy': np.mean(accuracy_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'F1-score': np.mean(f1_scores),
        'AUC': np.mean(roc_auc_scores)
    }

# Print the average cross-validation results for each model
if results:
    print("\nAverage Cross-Validation Results:")
    for model_name, scores in results.items():
        print(f"\n{model_name}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
else:
    print("\nCross-Validation was not performed because the dataset was not loaded.")

# ---  Same models but other method ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])
X_scaled = pd.DataFrame(X_scaled, columns=numerical_cols)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


if X_train_scaled is not None and y_train is not None:
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42) # הסרנו probability=True אם רק מדדים אחרים חשובים
    }

    n_splits_majority = 5
    results = {}

    for model_name, model in models.items():
        print(f"\nPerforming Multi-Part Balanced Training for {model_name}:")
        all_fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Cross-Validation חיצוני

        for outer_fold, (train_outer_index, val_outer_index) in enumerate(skf_outer.split(X_train_scaled, y_train)):
            X_train_outer = X_train_scaled.iloc[train_outer_index].copy()
            X_val_outer = X_train_scaled.iloc[val_outer_index].copy()
            y_train_outer = y_train.iloc[train_outer_index].copy()
            y_val_outer = y_train.iloc[val_outer_index].copy()

            X_train_mg = X_train_outer[y_train_outer == 'Mg']
            y_train_mg = y_train_outer[y_train_outer == 'Mg']
            X_train_mn = X_train_outer[y_train_outer == 'Mn']
            y_train_mn = y_train_outer[y_train_outer == 'Mn']

            n_mg_per_split = len(X_train_mg) // n_splits_majority
            mg_splits_indices = np.array_split(X_train_mg.index, n_splits_majority)
            fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for i in range(n_splits_majority):
                print(f"\n--- Outer Fold {outer_fold + 1}, Training Part {i + 1} ---")
                current_mg_indices = mg_splits_indices[i]
                X_train_mg_part = X_train_outer.loc[current_mg_indices]
                y_train_mg_part = y_train_outer.loc[current_mg_indices]

                X_train_balanced = pd.concat([X_train_mg_part, X_train_mn])
                y_train_balanced = pd.concat([y_train_mg_part, y_train_mn])

                # ערבוב הנתונים המאוזנים
                combined_train = pd.concat([X_train_balanced, y_train_balanced], axis=1).sample(frac=1, random_state=42)
                X_train_balanced = combined_train.drop('label', axis=1)
                y_train_balanced = combined_train['label']

                # --- Feature Selection ---
                selector = SelectKBest(mutual_info_classif, k=10)
                selector.fit(X_train_balanced, y_train_balanced)
                mi_scores = pd.Series(selector.scores_, index=X_train_balanced.columns)
                print(f"\n  Outer Fold {outer_fold + 1}, Part {i + 1} - Mutual Information Scores:")
                print(mi_scores.sort_values(ascending=False).head(10))
                selected_features_indices = selector.get_support(indices=True)
                selected_features = X_train_balanced.columns[selected_features_indices]
                print(f"\n  Outer Fold {outer_fold + 1}, Part {i + 1} - Selected Features: {selected_features}")
                X_train_selected = X_train_balanced[selected_features]
                X_val_selected = X_val_outer[selected_features]
                # --- End of Feature Selection ---

                model.fit(X_train_selected, y_train_balanced)
                y_pred_fold = model.predict(X_val_selected)

                accuracy = accuracy_score(y_val_outer, y_pred_fold)
                precision = precision_score(y_val_outer, y_pred_fold, average='binary', pos_label='Mg', zero_division=0)
                recall = recall_score(y_val_outer, y_pred_fold, average='binary', pos_label='Mg', zero_division=0)
                f1 = f1_score(y_val_outer, y_pred_fold, average='binary', pos_label='Mg', zero_division=0)

                print(f"  Outer Fold {outer_fold + 1}, Part {i + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

                fold_metrics['accuracy'].append(accuracy)
                fold_metrics['precision'].append(precision)
                fold_metrics['recall'].append(recall)
                fold_metrics['f1'].append(f1)

            for metric in all_fold_metrics:
                all_fold_metrics[metric].append(np.mean(fold_metrics[metric]))

        results[model_name] = {
            'Accuracy': np.mean(all_fold_metrics['accuracy']),
            'Precision': np.mean(all_fold_metrics['precision']),
            'Recall': np.mean(all_fold_metrics['recall']),
            'F1-score': np.mean(all_fold_metrics['f1'])
        }

    print("\nAverage Performance Metrics after Multi-Part Balanced Training (Cross-Validation):")
    for model_name, avg_metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")

else:
    print("\nData not loaded properly.")



# Separate features (X) and labels (y)
X = df_final.drop('label', axis=1)
y = df_final['label']
class_names = np.unique(y) # Get unique class names for printing
positive_label = 'Mg' # Define the positive label for ROC curve

# 4. Identify numerical columns
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# --- Handle missing values ---
imputer = SimpleImputer(strategy='median') # Using median as discussed
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
print("\nMissing values in numerical columns imputed using the median.")
# --- End of handling missing values ---

# 5. Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scaling using StandardScaler on numerical columns only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 7. Cross-Validation with StratifiedKFold for multiple models
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42) # probability=True for ROC AUC
}

results = {}
for model_name in models:
    results[model_name] = {} # אתחול מילון ריק עבור כל מודל

plt.figure(figsize=(12, 10))
for model_name, model in models.items():
    print(f"\nPerforming Cross-Validation for {model_name}:")
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_scaled, y_train)):
        X_train_fold_scaled = X_train_scaled.iloc[train_index]
        X_val_fold_scaled = X_train_scaled.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        # Train the model
        model.fit(X_train_fold_scaled, y_train_fold)

        # Make predictions on the validation set
        y_pred_proba_fold = model.predict_proba(X_val_fold_scaled)[:, 1] # Probability for the positive class

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_val_fold == positive_label, y_pred_proba_fold)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = roc_auc_score(y_val_fold == positive_label, y_pred_proba_fold)
        aucs.append(roc_auc)
        print(f"  Fold {fold + 1} AUC: {roc_auc:.4f}")
        plt.plot(fpr, tpr, alpha=0.2, label=f'{model_name} Fold {fold+1} (AUC = {roc_auc:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b' if model_name == 'Random Forest' else 'r', linestyle='-',
              label=f'Mean ROC {model_name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2)

    results[model_name]['AUC'] = mean_auc # Update the AUC in the results dictionary

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for All Folds and Mean')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print the average cross-validation results for each model
print("\nAverage Cross-Validation Results:")
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

else:
  print("\nCross-Validation was not performed because the dataset was not loaded.")
