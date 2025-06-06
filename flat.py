import pandas as pd

# Ensure access to project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load Excel
import pandas as pd

file_path = "full_step4.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Group by each step
grouped_df1 = df.groupby('group_id_step1').agg(lambda x: list(x)).reset_index()
grouped_df2 = df.groupby('group_id_step2').agg(lambda x: list(x)).reset_index()
grouped_df3 = df.groupby('group_id_step3').agg(lambda x: list(x)).reset_index()
grouped_df4 = df.groupby('group_id_step4').agg(lambda x: list(x)).reset_index()

# Save all to Excel with unique sheet names
with pd.ExcelWriter("reconciliation_outputs_factory_normalisation.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="df", index=False)
    grouped_df1.to_excel(writer, sheet_name="step1", index=False)
    grouped_df2.to_excel(writer, sheet_name="step2", index=False)  # corrected sheet name
    grouped_df3.to_excel(writer, sheet_name="step3", index=False)
    grouped_df4.to_excel(writer, sheet_name="step4", index=False)

