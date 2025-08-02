import pandas as pd
import numpy as np

def load_and_pivot(file_path, pivot_col, value_col, index_cols, special_handling=None):
    """Load CSV and pivot with robust column handling"""
    df = pd.read_csv(file_path)
    
    # Special handling for Migration data
    if special_handling == "migration":
        if "Gender" in df.columns:
            df = df.drop(columns=["Gender"])
    
    # Verify columns exist
    for col in [pivot_col, value_col] + index_cols:
        if col not in df.columns:
            available_cols = ", ".join(df.columns)
            raise ValueError(f"Column '{col}' not found. Available columns: {available_cols}")
    
    # Pivot with error handling
    try:
        pivoted = df.pivot_table(
            index=index_cols,
            columns=pivot_col,
            values=value_col,
            aggfunc='first'
        ).reset_index()
        
        # Clean column names
        pivoted.columns = [str(col).replace(" ", "_").replace("/", "_") for col in pivoted.columns]
        return pivoted
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Load and process all datasets
water = load_and_pivot(
    "Access to improved source of drinking water.csv",
    pivot_col="Sub Indicator",
    value_col="Value",
    index_cols=["State", "Sector"]
)

media = load_and_pivot(
    "Access to Mass Media and Broadband.csv",
    pivot_col="Internet Access",
    value_col="Value",
    index_cols=["State", "Sector"]
)

latrine = load_and_pivot(
    "Improved latrine and hand washing facilities within household.csv",
    pivot_col="Sub Indicator",
    value_col="Value",
    index_cols=["State", "Sector"]
)

assets = load_and_pivot(
    "Household Assets.csv",
    pivot_col="Sub Indicator",
    value_col="Value",
    index_cols=["State", "Sector"]
)

migration = load_and_pivot(
    "Main reason for Migration.csv",
    pivot_col="Main reason for Migration",
    value_col="Value",
    index_cols=["State", "Sector"],
    special_handling="migration"
)

mobile = load_and_pivot(
    "Usage of mobile phone.csv",
    pivot_col="Indicator",
    value_col="Value",
    index_cols=["State", "Sector"]
)

# Merge all datasets
merged = water.merge(media, on=["State", "Sector"], how="left") \
             .merge(latrine, on=["State", "Sector"], how="left") \
             .merge(assets, on=["State", "Sector"], how="left") \
             .merge(migration, on=["State", "Sector"], how="left") \
             .merge(mobile, on=["State", "Sector"], how="left")

# ===== SDG ENHANCEMENTS =====
# 1. Encode sector (0=Rural, 1=Urban)
merged["Sector"] = merged["Sector"].map({"Rural": 0, "Urban": 1})

# 2. SDG 6.1 compliance (90% threshold)
merged["SDG_6_Status"] = np.where(
    merged["Improved_Source_of_Drinking_Water"] >= 90,
    "Met", 
    "Not Met"
)

# 3. Urban-rural disparity calculation
def calculate_disparity(group):
    urban_mean = group[group["Sector"] == 1]["Improved_Source_of_Drinking_Water"].mean()
    rural_mean = group[group["Sector"] == 0]["Improved_Source_of_Drinking_Water"].mean()
    return urban_mean - rural_mean

merged["Urban_Rural_Gap"] = merged.groupby("State").apply(calculate_disparity).reset_index(level=0, drop=True)

# 4. Priority ranking (quintiles)
merged["Priority_Rank"] = pd.qcut(
    merged.groupby("State")["Improved_Source_of_Drinking_Water"].transform('mean'),
    q=5,
    labels=["Tier 1 (Urgent)", "Tier 2", "Tier 3", "Tier 4", "Tier 5 (Best)"]
)

# 5. Clean missing values (keep rows with target variable)
merged = merged.dropna(subset=["Improved_Source_of_Drinking_Water"])
# ===== END ENHANCEMENTS =====

# Save final dataset
merged.to_csv("final_sdg_water_data.csv", index=False)
print("✅ SDG-enhanced data saved to 'final_sdg_water_data.csv'")
print("\nSample rows:")
print(merged[["State", "Sector", "Improved_Source_of_Drinking_Water", 
             "SDG_6_Status", "Urban_Rural_Gap", "Priority_Rank"]].head())

print("\nKey Statistics:")
print(f"- SDG Compliance: {merged['SDG_6_Status'].value_counts().to_dict()}")
print(f"- Average Urban-Rural Gap: {merged['Urban_Rural_Gap'].mean():.1f}%")