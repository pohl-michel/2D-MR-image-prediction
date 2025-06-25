import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings("ignore")

# For shortening sizes in Python
METHODS = {
    "RNN with RTRL": "RTRL",
    "RNN with UORO": "UORO",
    "RNN with SnAp1": "SnAp1",
    "RNN with DNI": "DNI",
    "RNN with fixed W": "Fixed W",
    "sequence-specific transformer": "Seq-Transformer",
    "population transformer": "Pop-Transformer",
    "No prediction (lastest PCA weight)": "Prev. weight",
    "multivariate LMS": "LMS",
    "Previous image as prediction": "Prev. image",
}


def cohens_d(x, y):
    """
    Calculate Cohen's d for effect size between two groups.
    For paired samples (like in Wilcoxon), we use the difference divided by the pooled standard deviation.
    """
    # Remove NaN values from both arrays
    valid_indices = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_indices]
    y_clean = y[valid_indices]

    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan

    # Calculate the difference
    diff = x_clean - y_clean

    # Cohen's d for paired samples: mean difference / standard deviation of differences
    if np.std(diff, ddof=1) == 0:
        return np.nan

    return np.mean(diff) / np.std(diff, ddof=1)


def median_diff(x, y):
    """
    Calculate the median difference between two groups.
    This is useful for paired samples to understand the central tendency of differences.
    """
    # Remove NaN values from both arrays
    valid_indices = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_indices]
    y_clean = y[valid_indices]

    if len(x_clean) < 2 or len(y_clean) < 2:
        return np.nan

    # Calculate the median difference
    return np.median(x_clean - y_clean)


def load_excel_data(file_path, sheet_name, top_left_cell, bottom_right_cell):
    """
    Load data from Excel file within specified cell range.

    Parameters:
    - file_path: path to Excel file
    - sheet_name: name of the worksheet
    - top_left_cell: e.g., 'B1148'
    - bottom_right_cell: e.g., 'K1235'
    """
    # Read the entire sheet first
    df_full = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Parse cell references
    def parse_cell_ref(cell_ref):
        col_letters = "".join([c for c in cell_ref if c.isalpha()])
        row_num = int("".join([c for c in cell_ref if c.isdigit()]))

        # Convert column letters to number (A=1, B=2, etc.)
        col_num = 0
        for char in col_letters:
            col_num = col_num * 26 + (ord(char.upper()) - ord("A") + 1)

        return row_num - 1, col_num - 1  # Convert to 0-based indexing

    start_row, start_col = parse_cell_ref(top_left_cell)
    end_row, end_col = parse_cell_ref(bottom_right_cell)

    # Extract the data within the specified range
    data_subset = df_full.iloc[start_row : end_row + 1, start_col : end_col + 1]

    return data_subset


def process_forecasting_data(file_path, sheet_name, top_left_cell, bottom_right_cell, target_metric):
    """
    Process the forecasting data to extract method comparisons for a specific metric.

    Parameters:
    - file_path: path to Excel file
    - sheet_name: name of the worksheet
    - top_left_cell: e.g., 'B1148'
    - bottom_right_cell: e.g., 'K1235'
    - target_metric: metric to analyze (e.g., 'cross-correlation', 'nRMSE', 'SSIM', 'mean DVF error', 'max DVF error')
    """

    # Load data
    data = load_excel_data(file_path, sheet_name, top_left_cell, bottom_right_cell)

    # Set column names based on the structure we observed
    # Column 0: Method names, Column 1: Metrics, Columns 2+: Sequence data
    data.columns = ["Method", "Metric"] + [f"Seq_{i}" for i in range(len(data.columns) - 2)]

    # Clean method names (forward fill for empty cells)
    data["Method"] = data["Method"].fillna(method="ffill")

    # Shorten method names
    data["Method"].replace(METHODS, inplace=True)

    # Filter out confidence rows and focus on the target metric
    target_rows = data[
        (data["Metric"] == target_metric) & (~data["Metric"].str.contains("confidence", na=False))
    ].copy()

    # Replace placeholder values with NaN for missing data
    sequence_cols = [col for col in target_rows.columns if col.startswith("Seq_")]
    for col in sequence_cols:
        target_rows[col] = pd.to_numeric(target_rows[col], errors="coerce")

    # Create a clean DataFrame with methods as rows and sequences as columns
    methods_data = target_rows.set_index("Method")[sequence_cols]

    # Remove methods with all NaN values
    methods_data = methods_data.dropna(how="all")

    return methods_data


def perform_statistical_analysis(methods_data):
    """
    Perform pairwise statistical comparisons using Wilcoxon signed-rank test and Cohen's d.

    Parameters:
    - methods_data: DataFrame with methods as rows and sequences as columns

    Returns:
    - p_values_df: DataFrame with p-values from Wilcoxon tests
    - cohens_d_df: DataFrame with Cohen's d effect sizes
    """

    methods = methods_data.index.tolist()
    n_methods = len(methods)

    # Initialize result matrices
    p_values = np.full((n_methods, n_methods), np.nan)
    cohens_d_values = np.full((n_methods, n_methods), np.nan)
    median_diff_values = np.full((n_methods, n_methods), np.nan)

    # Perform pairwise comparisons
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                # Same method comparison
                p_values[i, j] = 1.0
                cohens_d_values[i, j] = 0.0
            else:
                # Get data for both methods
                data1 = methods_data.loc[method1].values
                data2 = methods_data.loc[method2].values

                # Find valid (non-NaN) paired observations
                valid_mask = ~(np.isnan(data1) | np.isnan(data2))

                if np.sum(valid_mask) >= 3:  # Need at least 3 paired observations
                    data1_clean = data1[valid_mask]
                    data2_clean = data2[valid_mask]

                    try:
                        # Wilcoxon signed-rank test (for paired samples)
                        # If all differences are zero, wilcoxon will raise an error
                        if not np.all(data1_clean == data2_clean):
                            statistic, p_value = wilcoxon(
                                data1_clean, data2_clean, alternative="two-sided", zero_method="wilcox"
                            )
                            p_values[i, j] = p_value
                        else:
                            p_values[i, j] = 1.0  # No difference

                        # Cohen's d
                        cohens_d_values[i, j] = cohens_d(data1_clean, data2_clean)

                        # Median difference
                        median_diff_values[i, j] = median_diff(data1_clean, data2_clean)

                    except ValueError as e:
                        # Handle cases where Wilcoxon test cannot be performed
                        print(f"Warning: Could not perform test for {method1} vs {method2}: {e}")

    # Create result DataFrames
    p_values_df = pd.DataFrame(p_values, index=methods, columns=methods)
    cohens_d_df = pd.DataFrame(cohens_d_values, index=methods, columns=methods)
    median_diff_df = pd.DataFrame(median_diff_values, index=methods, columns=methods)

    return p_values_df, cohens_d_df, median_diff_df


def create_combined_results_table(p_values_df, cohens_d_df, median_diff_df, significance_level=0.05):
    """
    Create a combined table with p-values and Cohen's d in the same cells.

    Parameters:
    - p_values_df: DataFrame with p-values
    - cohens_d_df: DataFrame with Cohen's d values
    - significance_level: threshold for statistical significance

    Returns:
    - combined_df: DataFrame with formatted results
    """

    methods = p_values_df.index.tolist()
    methods_index = methods[:-1]  # discarding the last row because it would be empty anyway
    combined_results = []

    for i, method1 in enumerate(methods_index):
        row_results = []
        for j, method2 in enumerate(methods):
            p_val = p_values_df.iloc[i, j]
            cohens_d_val = cohens_d_df.iloc[i, j]
            median_diff_val = median_diff_df.iloc[i, j]

            if i == j:
                # Diagonal elements
                cell_content = "—"
            elif i > j:
                cell_content = "-"  # Avoid duplicate entries in lower triangle
            elif np.isnan(p_val) or np.isnan(cohens_d_val):
                # Missing data
                cell_content = "N/A"
            else:
                # Format: p-value (Cohen's d)
                # Mark significant p-values with asterisk
                p_str = f"{p_val:.3f}"
                if p_val < significance_level:
                    p_str += "*"

                # check if median difference and cohen d have same sign
                if np.sign(median_diff_val) != np.sign(cohens_d_val):
                    p_str += "°"

                d_str = f"{cohens_d_val:.2f}"
                cell_content = f"{p_str} ({d_str})"

            row_results.append(cell_content)

        combined_results.append(row_results)

    combined_df = pd.DataFrame(combined_results, index=methods_index, columns=methods)

    return combined_df


# Main analysis function
def analyze_forecasting_methods(file_path, sheet_name, top_left_cell, bottom_right_cell, target_metric):
    """
    Complete analysis pipeline for forecasting method comparison.

    Parameters:
    - file_path: path to Excel file
    - sheet_name: name of the worksheet
    - top_left_cell: e.g., 'B1148'
    - bottom_right_cell: e.g., 'K1235'
    - target_metric: metric to analyze

    Returns:
    - Dictionary with results including raw data, p-values, Cohen's d, and combined table
    """

    print(f"Analyzing metric: {target_metric}")
    print(f"Loading data from {sheet_name}, range {top_left_cell}:{bottom_right_cell}")

    # Process data
    methods_data = process_forecasting_data(file_path, sheet_name, top_left_cell, bottom_right_cell, target_metric)

    print(f"Found {len(methods_data)} methods with data for {target_metric}")
    print("Methods:", list(methods_data.index))

    # Perform statistical analysis
    p_values_df, cohens_d_df, median_diff_df = perform_statistical_analysis(methods_data)

    # Create combined results table
    combined_table = create_combined_results_table(p_values_df, cohens_d_df, median_diff_df)

    # Summary statistics
    print(f"\nSummary for {target_metric}:")
    print(f"- Methods compared: {len(methods_data)}")
    print(f"- Sequences per method: {methods_data.shape[1]}")
    print(
        f"- Significant comparisons (p < 0.05): {(p_values_df < 0.05).sum().sum() - len(methods_data)}"
    )  # Subtract diagonal

    results = {
        "metric": target_metric,
        "raw_data": methods_data,
        "p_values": p_values_df,
        "median_diffs": median_diff_df,
        "cohens_d": cohens_d_df,
        "combined_table": combined_table,
        "summary_stats": {
            "n_methods": len(methods_data),
            "n_sequences": methods_data.shape[1],
            "n_significant": (p_values_df < 0.05).sum().sum() - len(methods_data),
        },
    }

    return results
